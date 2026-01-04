def get_active_window_name():
    """
    获取当前活动窗口的应用名称 (Windows)
    """
    try:
        import pygetwindow as gw
        active_window = gw.getActiveWindow()
        return active_window.title
    except ImportError:
        print("请先安装 pygetwindow 库: pip install pygetwindow")
        return None
    except Exception as e:
        print(f"获取窗口名称时出错: {e}")
        return None

def get_active_process_name():
    """
    获取当前活动窗口的进程名称 (Windows)
    """
    try:
        import win32gui
        import win32process
        import psutil
        
        # 获取前台窗口句柄
        hwnd = win32gui.GetForegroundWindow()
        
        # 获取窗口标题
        window_title = win32gui.GetWindowText(hwnd)
        
        # 获取进程ID
        thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        
        # 获取进程名称
        try:
            process = psutil.Process(process_id)
            process_name = process.name()
            return {
                'window_title': window_title,
                'process_name': process_name,
                'process_id': process_id
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                'window_title': window_title,
                'process_name': 'Unknown',
                'process_id': process_id
            }
    except ImportError:
        print("请安装 pywin32 和 psutil 库: pip install pywin32 psutil")
        return None
    except Exception as e:
        print(f"获取进程信息时出错: {e}")
        return None

def main():
    print("使用 pygetwindow 获取窗口标题:")
    window_name = get_active_window_name()
    if window_name:
        print(f"当前活动窗口: {window_name}")
    
    print("\n使用 win32api 获取详细信息:")
    process_info = get_active_process_name()
    if process_info:
        print(f"窗口标题: {process_info['window_title']}")
        print(f"进程名称: {process_info['process_name']}")
        print(f"进程ID: {process_info['process_id']}")

if __name__ == "__main__":
    main()