import pygetwindow as gw
import time

# # 获取所有窗口的标题
# titles = gw.getAllTitles()

# print("当前所有窗口标题：")
# for title in titles:
#     if title.strip():  # 过滤空标题
#         print(title)

cypnest_windows = gw.getWindowsWithTitle('Blender')
if cypnest_windows:
    cypnest_window = cypnest_windows[0]  # 获取第一个匹配的窗口
    print(f"激活窗口: {cypnest_window.title}")
    
   
    # 使用pywin32进行更精确的窗口控制（如果已安装）
    import win32gui
    import win32con
    
    hwnd = cypnest_window._hWnd
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 恢复窗口（如果最小化）
   
    
    print(f"窗口已放到最前端: {cypnest_window.title}")
   
else:
    print("未找到CypNest窗口")