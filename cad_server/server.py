# server.py
from pyautocad import Autocad
import win32gui
from flask import Flask, jsonify

# 初始化AutoCAD连接
acad = Autocad()
# 获取AutoCAD窗口句柄并激活
hwnd = acad.app.HWND
win32gui.SetForegroundWindow(hwnd)

# 创建Flask应用实例
app = Flask(__name__)


@app.route('/bring-to-front', methods=['GET'])
def bring_to_front():
    """
    将AutoCAD窗口置于前台
    
    Returns:
        JSON格式的操作结果
    """
    try:
        win32gui.SetForegroundWindow(hwnd)
        return jsonify({
            'status': 'success',
            'message': 'AutoCAD window brought to front'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # 启动Flask服务器，默认端口5100
    app.run(host='0.0.0.0', port=5100, debug=True)