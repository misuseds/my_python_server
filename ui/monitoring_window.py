from PyQt6.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject


class MonitoringWindow(QMainWindow):
    """监控显示窗口 - 用于显示VLM分析或吐槽内容"""

    # 定义信号用于线程间通信
    text_added = pyqtSignal(str)
    window_shown = pyqtSignal()

    def __init__(self, title: str, window_type: str):
        """
        初始化监控窗口

        Args:
            title: 窗口标题
            window_type: 窗口类型 ('analysis' 或 'commentary')
        """
        super().__init__()
        self.window_type = window_type
        self._setup_window(title)
        self._setup_ui()
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide)
        # 连接信号到槽
        self.text_added.connect(self._add_text_safe)
        self.window_shown.connect(self._show_safe)
        self.hide()

    def _setup_window(self, title: str):
        """设置窗口属性"""
        self.setWindowTitle(title)

        # 吐槽窗口使用无边框透明窗口
        if self.window_type == 'commentary':
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint
            )
            # 确保使用透明背景
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        else:
            self.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowStaysOnTopHint
            )
            # 确保使用透明背景
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置窗口位置和大小
        if self.window_type == 'commentary':
            # 吐槽窗口在左侧，避免与主窗口和VLM窗口重叠
            self.setGeometry(100, 100, 500, 200)

    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        # 隐藏垂直滚动条
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 隐藏水平滚动条
        self.text_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 根据窗口类型设置样式
        if self.window_type == 'analysis':
            color = '#ff0000'  # VLM分析窗口用红色
            bg_alpha = 0  # 完全透明背景
        else:
            color = '#ff0000'  # 吐槽窗口用红色
            bg_alpha = 0  # 完全透明背景

        # 使用字符串格式化，避免f-string中的语法错误
        style_sheet = """
            QTextEdit {
                background-color: rgba(30, 30, 30, %d);
                color: %s;
                border: none;
                font-family: Consolas, monospace;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            }
        """
        self.text_display.setStyleSheet(style_sheet % (bg_alpha, color))

        layout.addWidget(self.text_display)

    def add_text(self, text: str):
        """
        添加文本到窗口（只显示最新一条，5秒后隐藏）
        """
        # 使用信号在主线程中添加文本
        self.text_added.emit(text)

    def _add_text_safe(self, text: str):
        """
        在主线程中安全地添加文本到窗口
        """
        # 清空旧内容，只显示最新一条
        self.text_display.clear()
        self.text_display.setPlainText(text.strip())
        
        # 显示窗口
        super().show()
        
        # 重置隐藏定时器（10秒后隐藏）
        self.hide_timer.stop()
        self.hide_timer.start(10000)

    def clear_text(self):
        """清空窗口内容"""
        self.text_display.clear()

    def show(self):
        """
        安全地显示窗口（使用信号在主线程中调用）
        """
        self.window_shown.emit()

    def _show_safe(self):
        """
        在主线程中安全地显示窗口
        """
        super().show()
