# server.py
from pyautocad import Autocad
import win32gui
from flask import Flask, jsonify, request
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入删除区域功能
from delete_area import change_objects_color_by_window

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


@app.route('/delete-area', methods=['GET'])
def delete_area():
    """
    删除指定区域内的对象（GET方法便于调试）
    
    Query Parameters:
        min_x, min_y, min_z: 窗口左下角点坐标
        max_x, max_y, max_z: 窗口右上角点坐标
    
    Returns:
        JSON格式的操作结果
    """
    try:
        # 从查询参数获取坐标值，提供默认值
        min_x = float(request.args.get('min_x', 0))
        min_y = float(request.args.get('min_y', 0))
        min_z = 0
        max_x = float(request.args.get('max_x', 100))
        max_y = float(request.args.get('max_y', 100))
        max_z = 0
        
        # 构造坐标点
        min_point = (min_x, min_y, min_z)
        max_point = (max_x, max_y, max_z)
        
        # 执行删除操作
        count = change_objects_color_by_window(acad, min_point, max_point)
        
        return jsonify({
            'status': 'success',
            'message': f'Deleted {count} objects',
            'deleted_count': count,
            'area': {
                'min_point': min_point,
                'max_point': max_point
            }
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinate values. Please provide numeric values.'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # 启动Flask服务器，默认端口5100
    app.run(host='0.0.0.0', port=5100, debug=True)