import sys
import asyncio
import os
import json
import threading
import queue
import re
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QDialog, QScrollArea, QGridLayout, QMessageBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QKeySequence, QShortcut
import importlib.util
from PyQt6 import sip


def extract_content_from_response(data):
    """从 LLM 响应中安全提取纯文本内容，兼容 DeepSeek/OpenAI 格式"""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first_choice = choices[0]
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                elif content is None:
                    return ""
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content.strip()
            if "finish_reason" in first_choice:
                return ""
        content = data.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


class ToolsDialog(QDialog):
    def __init__(self, tools_by_server, all_tools_mapping=None, parent=None):
        super().__init__(parent)
        self.all_tools_mapping = all_tools_mapping or {}
        self.tools_by_server = tools_by_server
        self.setWindowTitle("可用的MCP工具")
        self.setGeometry(300, 300, 800, 600)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: 1px solid #555; background-color: #222; }")
        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        for server_name, tools in tools_by_server.items():
            server_title = QLabel(f"<b>{server_name}</b>")
            server_title.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #00ff41;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    padding: 5px;
                    border-bottom: 1px solid #555;
                }
            """)
            scroll_layout.addWidget(server_title)
            tools_vbox = QVBoxLayout()
            tools_vbox.setSpacing(8)
            tools_vbox.setContentsMargins(10, 0, 0, 0)
            for tool_name, tool_desc in tools.items():
                tool_number = self.find_tool_number(tool_name, server_name)
                if tool_number:
                    name_label = QLabel(f"• [{tool_number}] {tool_name}:")
                else:
                    name_label = QLabel(f"• {tool_name}:")
                name_label.setStyleSheet("QLabel { font-weight: bold; color: #ffffff; margin-left: 10px; }")
                desc_label = QLabel(tool_desc)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("QLabel { color: #cccccc; margin-left: 25px; margin-bottom: 8px; }")
                tools_vbox.addWidget(name_label)
                tools_vbox.addWidget(desc_label)
            scroll_layout.addLayout(tools_vbox)
        scroll_layout.addStretch()
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def find_tool_number(self, tool_name, display_server_name):
        original_server_name = self.get_original_server_name(display_server_name)
        for num, info in self.all_tools_mapping.items():
            if info['name'] == tool_name and (info['server'] == original_server_name or self.format_server_name(info['server']) == display_server_name):
                return num
        return None

    def get_original_server_name(self, display_server_name):
        if display_server_name.endswith(" 工具服务器"):
            formatted_name = display_server_name[:-5].strip()
            original = formatted_name.replace(' ', '-').lower()
            for candidate in [original, original + "-tool"]:
                for _, info in self.all_tools_mapping.items():
                    if info['server'] == candidate:
                        return candidate
            return original
        return display_server_name

    def format_server_name(self, original_server_name):
        display_name = original_server_name.replace('-tool', '').replace('-', ' ').title()
        display_name += " 工具服务器"
        return display_name


class StreamWorker(QObject):
    stream_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, llm_service, messages, parent_window, tools_schema=None):
        super().__init__()
        self.llm_service = llm_service
        self.messages = messages
        self.parent_window = parent_window
        self.tools_schema = tools_schema
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run_stream(self):
        try:
            for chunk in self.generate_stream():
                if self._is_stopped:
                    break
                self.stream_signal.emit(chunk)
            self.finished_signal.emit()
        except Exception as e:
            if not self._is_stopped:
                self.error_signal.emit(f"错误: {str(e)}")
            self.finished_signal.emit()

    def generate_stream(self):
        response = self.llm_service.create_stream(self.messages, tools=self.tools_schema)
        for chunk in response:
            if self._is_stopped:
                break
            content = extract_content_from_response(chunk)
            if content:
                yield content


class ToolLoader(QObject):
    tools_loaded = pyqtSignal(object)
    loading_failed = pyqtSignal(str)

    def __init__(self, mcp_client):
        super().__init__()
        self.mcp_client = mcp_client

    def load_tools(self):
        try:
            tools_by_server = {}
            all_tools_mapping = {}
            tool_counter = 1

            for server_name in self.mcp_client.servers.keys():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        tools = loop.run_until_complete(self.mcp_client.list_tools(server_name))
                        if tools:
                            server_tools = {}
                            for tool in tools:
                                server_tools[tool.name] = tool.description
                                all_tools_mapping[str(tool_counter)] = {
                                    'name': tool.name,
                                    'description': tool.description,
                                    'server': server_name,
                                    'input_schema': getattr(tool, 'inputSchema', {"type": "object", "properties": {}, "required": []}),
                                }
                                tool_counter += 1
                            display_server_name = server_name.replace('-tool', '').replace('-', ' ').title() + " 工具服务器"
                            tools_by_server[display_server_name] = server_tools
                        else:
                            display_server_name = server_name.replace('-tool', '').replace('-', ' ').title() + " 工具服务器"
                            tools_by_server[display_server_name] = {server_name: self.mcp_client.servers[server_name]['description']}
                            all_tools_mapping[str(tool_counter)] = {
                                'name': server_name,
                                'description': self.mcp_client.servers[server_name]['description'],
                                'server': server_name,
                                'input_schema': {"type": "object", "properties": {}, "required": []},
                            }
                            tool_counter += 1
                    finally:
                        loop.close()
                except Exception as e:
                    print(f"获取服务器 {server_name} 的工具列表失败: {str(e)}")
                    continue

            self.tools_loaded.emit((all_tools_mapping, tools_by_server))
        except Exception as e:
            self.loading_failed.emit(f"加载工具列表失败: {str(e)}")


class MCPAICaller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.output_buffer = ""
        self.worker_thread = None
        self.worker = None
        self.tool_loader_thread = None
        self.tool_loader = None
        self.is_loading_tools = False
        self.all_tools_mapping = {}
        self.tools_by_server = {}
        self.loading_dialog = None
        self.pending_show_tools = False

        self.setup_window()
        self.initialize_services()
        self.initialize_clients()
        self.setup_ui()
        self.setup_shortcuts()

    def setup_window(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(200, 200, 400, 200)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 0))
        self.setPalette(palette)

    def initialize_services(self):
        llm_spec = importlib.util.spec_from_file_location(
            "llm_class",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_server", "llm_class.py")
        )
        llm_module = importlib.util.module_from_spec(llm_spec)
        llm_spec.loader.exec_module(llm_module)
        self.LLMService = llm_module.LLMService
        self.llm_service = self.LLMService()

    def initialize_clients(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_client_path = os.path.join(current_dir, "mcp_client.py")
        spec = importlib.util.spec_from_file_location("mcp_client", mcp_client_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        self.mcp_client = mcp_module.MCPClient()

        self.tool_loader = ToolLoader(self.mcp_client)
        self.tool_loader.tools_loaded.connect(self.on_tools_loaded)
        self.tool_loader.loading_failed.connect(self.on_tools_loading_failed)
        self.async_refresh_tools_list()

    def on_tools_loaded(self, data_tuple):
        all_tools_mapping, tools_by_server = data_tuple
        self.all_tools_mapping = all_tools_mapping
        self.tools_by_server = tools_by_server
        self.is_loading_tools = False
        if self.loading_dialog:
            self.loading_dialog.accept()
            self.loading_dialog = None
        if self.pending_show_tools:
            self.pending_show_tools = False
            self._show_tools_dialog_now()

    def on_tools_loading_failed(self, error_msg):
        self.is_loading_tools = False
        if self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("错误")
        error_dialog.setText(f"加载工具列表失败: {error_msg}")
        error_dialog.setIcon(QMessageBox.Icon.Warning)
        error_dialog.exec()
        self.pending_show_tools = False

    def async_refresh_tools_list(self):
        if self.is_loading_tools and self.tool_loader_thread and self.tool_loader_thread.isRunning():
            return
        self.is_loading_tools = True
        if self.tool_loader_thread and self.tool_loader_thread.isRunning():
            self.tool_loader_thread.quit()
            self.tool_loader_thread.wait(2000)
        self.tool_loader_thread = QThread()
        self.tool_loader.moveToThread(self.tool_loader_thread)
        self.tool_loader_thread.started.connect(self.tool_loader.load_tools)
        self.tool_loader_thread.finished.connect(self.tool_loader_thread.deleteLater)
        self.tool_loader_thread.start()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.caption_text = QTextEdit()
        self.caption_text.setReadOnly(True)
        self.caption_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0);
                color: #00ff41;
                border: none;
                font-family: Consolas, monospace;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.caption_text.setMaximumHeight(120)
        layout.addWidget(self.caption_text)

        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("输入消息...")
        self.input_text.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 180);
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #00ff41;
            }
        """)
        self.input_text.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_text)
        self.input_text.setFocus()
        self.drag_position = None

    def setup_shortcuts(self):
        esc_shortcut = QShortcut(QKeySequence('Escape'), self)
        esc_shortcut.activated.connect(self.close)
        quit_shortcut = QShortcut(QKeySequence('Ctrl+Q'), self)
        quit_shortcut.activated.connect(self.close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None

    def clear_captions(self):
        self.caption_text.clear()

    def add_caption_line(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return
        current = self.caption_text.toPlainText()
        new = current + "\n" + text if current else text
        lines = new.split('\n')[-20:]
        self.caption_text.setPlainText('\n'.join(lines))
        self.caption_text.moveCursor(self.caption_text.textCursor().MoveOperation.End)
        QTimer.singleShot(30000, self.clear_captions)

    def send_message(self):
        user_input = self.input_text.text().strip()
        if not user_input:
            return
        self.input_text.clear()

        if user_input == '/h':
            self.show_tools_dialog()
            return

        if user_input.startswith('/r ') and len(user_input) > 3:
            cmd = user_input[3:].strip()
            if cmd.isdigit():
                self.handle_run_command_by_index(int(cmd))
            else:
                self.process_message_with_function_call(cmd)
            return

        self.process_message_with_function_call(user_input)

    def get_mcp_tools_schema(self):
        if not self.all_tools_mapping:
            return []
        tools = []
        for tool_id, tool_info in self.all_tools_mapping.items():
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_info['name'],
                    "description": tool_info['description'],
                    "parameters": tool_info.get('input_schema', {"type": "object", "properties": {}, "required": []})
                }
            }
            tools.append(tool_schema)
        return tools

    def handle_run_command_by_index(self, index):
        if str(index) not in self.all_tools_mapping:
            self.add_caption_line(f"错误：没有找到编号为 {index} 的工具")
            return

        tool_info = self.all_tools_mapping[str(index)]
        tools_schema = self.get_mcp_tools_schema()
        messages = [{"role": "user", "content": f"请立即调用工具 '{tool_info['name']}'。"}]

        try:
            result = self.llm_service.create(messages, tools=tools_schema)
            self.process_function_call_response(result, messages)
        except Exception as e:
            self.add_caption_line(f"调用失败: {str(e)}")

    def process_function_call_response(self, result, original_messages):
        try:

            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                tool_calls = choice.get('message', {}).get('tool_calls', [])
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call['function']['name']
                        arguments = json.loads(tool_call['function']['arguments'])

                        server_name = None
                        for tool_id, info in self.all_tools_mapping.items():
                            if info['name'] == function_name:
                                server_name = info['server']
                                break

                        if not server_name:
                            self.add_caption_line(f"错误：找不到工具 {function_name} 对应的服务器")
                            continue

                        self.add_caption_line(f"[调用工具] {function_name}")
                        print(f"【MCP CALL】Calling {server_name}.{function_name} with args: {arguments}")

                        # 执行 MCP 调用（现在能获取真实结果！）
                        mcp_result = self.execute_mcp_call_sync(server_name, function_name, arguments)
                        print(f"【MCP RESULT】{mcp_result}")

                        # 显示错误（如果存在）
                        if isinstance(mcp_result, dict) and "error" in mcp_result:
                            self.add_caption_line(f"[错误] {mcp_result['error']}")

                        # 构造 tool response 消息
                        updated_messages = original_messages.copy()
                        updated_messages.append({
                            "role": "assistant",
                            "tool_calls": [tool_call]
                        })
                        updated_messages.append({
                            "role": "tool",
                            "content": json.dumps(mcp_result, ensure_ascii=False),
                            "tool_call_id": tool_call.get('id', '')
                        })

                        # 获取最终自然语言回复
                        final_result = self.llm_service.create(updated_messages)
                        final_content = extract_content_from_response(final_result)
                        self.add_caption_line(final_content if final_content else "[AI未返回内容]")
                else:
                    content = extract_content_from_response(result)
                    self.add_caption_line(content if content else "[AI未返回内容]")
            else:
                content = extract_content_from_response(result)
                self.add_caption_line(content if content else "[AI未返回内容]")
        except Exception as e:
            import traceback
            error_msg = f"处理函数调用时出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.add_caption_line(f"处理函数调用时出错: {str(e)}")

    def show_tools_dialog(self):
        if self.tools_by_server:
            self._show_tools_dialog_now()
        else:
            self.pending_show_tools = True
            if not self.is_loading_tools:
                self.async_refresh_tools_list()
            if not self.loading_dialog:
                self.loading_dialog = QMessageBox(self)
                self.loading_dialog.setWindowTitle("加载中")
                self.loading_dialog.setText("正在加载MCP工具列表，请稍候...")
                self.loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
                self.loading_dialog.show()

    def _show_tools_dialog_now(self):
        if self.tools_by_server:
            dialog = ToolsDialog(self.tools_by_server, self.all_tools_mapping, self)
            dialog.exec()
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("暂无可用的MCP工具")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.exec()

    def _cleanup_worker_thread(self):
        if self.worker_thread and self.worker_thread.isRunning():
            if self.worker and hasattr(self.worker, 'stop'):
                self.worker.stop()
            self.worker_thread.quit()
            if not self.worker_thread.wait(5000):
                print("警告: 流式传输线程未能在规定时间内退出")
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def process_message_with_function_call(self, user_input):
        messages = [{"role": "user", "content": user_input}]
        tools_schema = self.get_mcp_tools_schema() if self.tools_by_server else None

        try:
            result = self.llm_service.create(messages, tools=tools_schema)
            choices = result.get("choices", [])
            if choices and "tool_calls" in choices[0].get("message", {}):
                self.process_function_call_response(result, messages)
            else:
                content = extract_content_from_response(result)
                if content:
                    self.add_caption_line(content)
        except Exception as e:
            self.add_caption_line(f"处理消息时出错: {str(e)}")

    def update_output_buffer(self, content_chunk):
        if not isinstance(content_chunk, str):
            return
        self.output_buffer += content_chunk
        if any(c in self.output_buffer for c in '.!?\n。！？') or len(self.output_buffer) >= 50:
            sentences = re.split(r'([.!?。！？\n]+)', self.output_buffer)
            display_parts = []
            i = 0
            while i < len(sentences):
                part = sentences[i]
                if i + 1 < len(sentences):
                    part += sentences[i + 1]
                    i += 2
                else:
                    i += 1
                if part.strip():
                    display_parts.append(part)
            if display_parts:
                self.add_caption_line(''.join(display_parts))
                self.output_buffer = ''

    def on_stream_finished(self):
        if self.output_buffer.strip():
            try:
                full_response = json.loads(self.output_buffer)
                self.process_function_call_response(full_response, [{"role": "user", "content": "dummy"}])
            except json.JSONDecodeError:
                self.add_caption_line(self.output_buffer)
        self.output_buffer = ""

    def on_stream_error(self, error_msg):
        self.add_caption_line(error_msg)
        self.output_buffer = ""

    # ==============================
    # 🔧 关键修复：正确同步执行 MCP 调用
    # ==============================
    def execute_mcp_call_sync(self, server_name, tool_name, arguments):
        """同步执行 MCP 调用，使用 Queue 获取子线程结果"""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.mcp_client.call_tool(server_name, tool_name, arguments)
                )
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
            finally:
                loop.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=15)  # 给 Blender 等慢启动工具留足时间

        if thread.is_alive():
            return {"error": "MCP 调用超时（15秒）"}

        if not exception_queue.empty():
            exc = exception_queue.get()
            return {"error": f"MCP 执行异常: {str(exc)}"}

        if not result_queue.empty():
            result = result_queue.get()
            # 尝试从 MCP Result 提取文本内容
            try:
                texts = []
                for item in getattr(result, 'content', []):
                    if hasattr(item, 'text') and isinstance(item.text, str):
                        texts.append(item.text.strip())
                if texts:
                    return {"result": "\n".join(texts)}
                else:
                    return {"result": str(result)}
            except Exception as parse_err:
                return {"result": str(result), "warning": f"解析结果时出错: {parse_err}"}
        else:
            return {"error": "MCP 调用无返回结果"}

    def closeEvent(self, event):
        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
        self._cleanup_worker_thread()
        if self.tool_loader_thread and self.tool_loader_thread.isRunning():
            self.tool_loader_thread.quit()
            self.tool_loader_thread.wait(3000)
        self.is_loading_tools = False
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QMainWindow { background-color: transparent; }")
    window = MCPAICaller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()