import win32gui
import win32ui
import win32con
import win32api
from PIL import Image
import time
from ctypes import windll

def capture_window_by_title(window_title, output_path="window_capture.png"):
    """
    根据窗口标题截取指定窗口内容，无需激活窗口
    
    Args:
        window_title: 窗口标题（部分匹配）
        output_path: 输出图片路径
    """
    # 查找窗口句柄
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if window_title.lower() in window_text.lower():
                windows.append(hwnd)
        return True

    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)

    if not windows:
        print(f"未找到包含 '{window_title}' 的窗口")
        return None

    # 选择第一个匹配的窗口
    hwnd = windows[0]
    
    # 获取窗口位置和大小
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # 获取设备上下文
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 创建位图
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 使用 PrintWindow API 截取窗口
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    
    if result:
        # 转换为PIL图像
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        im.save(output_path)
        

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    
    return output_path if result else None


# 使用示例
if __name__ == "__main__":
    # 截取Edge浏览器窗口
    capture_window_by_title("Edge", "edge_capture.png")