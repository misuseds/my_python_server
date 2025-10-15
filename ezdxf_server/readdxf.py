import ezdxf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
from flask import Flask, request, jsonify, send_file, render_template_string
import os
import subprocess
import tempfile

app = Flask(__name__)

import subprocess
import os

def convert_dwg_to_dxf(dwg_file_path):
    """
    使用ODA File Converter将DWG文件转换为DXF文件
    
    Args:
        dwg_file_path (str): DWG文件路径
        
    Returns:
        dict: 转换结果，包含状态和DXF文件路径或错误信息
    """
    try:
        # 检查DWG文件是否存在
        if not os.path.exists(dwg_file_path):
            return {
                "status": "error",
                "message": f"DWG文件不存在: {dwg_file_path}"
            }

        # 获取文件目录和文件名
        dwg_dir = os.path.dirname(dwg_file_path)
        dwg_filename = os.path.basename(dwg_file_path)
        dxf_filename = dwg_filename.replace('.dwg', '.dxf')
        dxf_file_path = dwg_file_path.replace('.dwg', '.dxf')
        if dxf_file_path == dwg_file_path:
            dxf_file_path = f"{dwg_file_path}.dxf"

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(dxf_file_path)
        os.makedirs(output_dir, exist_ok=True)

        # 参数定义（按官方规范）
        TEIGHA_PATH = "ODAFileConverter"
        INPUT_FOLDER = dwg_dir
        OUTPUT_FOLDER = output_dir
        OUTVER = "ACAD2018"           # 输入版本（如原文件是 ACAD2018）
        OUTFORMAT = "DXF"             # 输出格式：只能是 DXF / DWG / DXB
        RECURSIVE = "0"
        AUDIT = "1"
        INPUTFILTER = "*.DWG"         # 只处理DWG文件

        # 构建命令（注意顺序！）
        cmd = [
            TEIGHA_PATH,
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            OUTVER,           # 输入版本
            OUTFORMAT,        # 输出格式 ← 必须是 DXF/DWG/DXB
            RECURSIVE,
            AUDIT,
            INPUTFILTER
        ]

        print(f"执行命令: {' '.join(cmd)}")

        # 执行转换命令
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # 打印调试信息
        print("ODA 输出:", result.stdout)
        print("ODA 错误:", result.stderr)

        if result.returncode == 0:
            # 检查DXF文件是否生成
            if os.path.exists(dxf_file_path):
                # 验证DXF文件是否有效
                if is_valid_dxf(dxf_file_path):
                    return {
                        "status": "success",
                        "message": f"DWG文件已成功转换为DXF: {dxf_file_path}",
                        "dxf_path": dxf_file_path
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"生成的DXF文件结构不完整或损坏: {dxf_file_path}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"转换完成但未生成DXF文件。请检查输出目录: {output_dir}"
                }
        else:
            return {
                "status": "error",
                "message": f"ODA转换失败: {result.stderr}"
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "转换超时"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "未找到ODAFileConverter命令，请确保ODA File Converter已正确安装并加入PATH"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"转换过程中发生错误: {str(e)}"
        }

def is_valid_dxf(dxf_path):
    """检查DXF文件是否有效"""
    try:
        doc = ezdxf.readfile(dxf_path)
        return True
    except Exception as e:
        print(f"DXF文件校验失败: {e}")
        return False


def process_dxf_file(dxf_file_path):
    """
    处理DXF文件并生成图像
    
    Args:
        dxf_file_path (str): DXF文件路径
        
    Returns:
        dict: 处理结果
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(dxf_file_path):
            return {
                "status": "error",
                "message": f"文件不存在: {dxf_file_path}"
            }

        # 加载DXF文件
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()  # 获取模型空间

        # 创建图形对象
        fig, ax = plt.subplots()

        lines = []

        for entity in msp:
            if entity.dxftype() == 'LINE':
                lines.append([(entity.dxf.start.x, entity.dxf.start.y),
                              (entity.dxf.end.x, entity.dxf.end.y)])
            elif entity.dxftype() == 'SPLINE':
                fit_points = entity.control_points
                lines.append([(fit_points[0][0], fit_points[0][1]),
                              (fit_points[-1][0], fit_points[-1][1])])
            elif entity.dxftype() == 'ARC':
                arc = entity
                center = (arc.dxf.center.x, arc.dxf.center.y)
                radius = arc.dxf.radius
                start_angle = arc.dxf.start_angle
                end_angle = arc.dxf.end_angle
                if start_angle > end_angle:
                    end_angle += 360
                arc_patch = Arc(center, 2*radius, 2*radius, angle=0,
                                theta1=start_angle, theta2=end_angle,
                                color='blue', linewidth=0.5)
                ax.add_patch(arc_patch)
            else:
                print('未处理的实体类型:', entity.dxftype())

        if lines:
            line_segments = LineCollection(lines, linewidths=0.5, colors='blue')
            ax.add_collection(line_segments)

            # 设置坐标范围
            x_coords = [p[0] for l in lines for p in l]
            y_coords = [p[1] for l in lines for p in l]
            ax.set_xlim(min(x_coords) - 5, max(x_coords) + 5)
            ax.set_ylim(min(y_coords) - 5, max(y_coords) + 5)

        plt.gca().set_aspect('equal', adjustable='box')
        output_path = f"{dxf_file_path}.png"
        plt.savefig(output_path)
        plt.close(fig)

        return {
            "status": "success",
            "message": f"DXF文件已处理并保存为 {output_path}",
            "output_path": output_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.route('/process_dxf', methods=['GET'])
def handle_process_dxf():
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({"status": "error", "message": "缺少dxf_path参数"}), 400

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"文件不存在: {file_path}"}), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return jsonify(conversion_result), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return jsonify({
            "status": "error",
            "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
        }), 400

    result = process_dxf_file(dxf_file_path)
    status_code = 200 if result["status"] == "success" else 500
    return jsonify(result), status_code


@app.route('/view_dxf', methods=['GET'])
def view_dxf():
    file_path = request.args.get('dxf_path')
    if not file_path:
        return render_template_string(HTML_TEMPLATE, title="错误", message="缺少dxf_path参数", image_url=None), 400

    if not os.path.exists(file_path):
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"文件不存在: {file_path}", image_url=None), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return render_template_string(HTML_TEMPLATE, title="转换失败", message=conversion_result["message"], image_url=None), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。", image_url=None), 400

    result = process_dxf_file(dxf_file_path)
    if result["status"] == "success":
        image_path = result["output_path"]
        image_url = f"/image?path={image_path}"
        return render_template_string(HTML_TEMPLATE, title="文件处理结果", message=result["message"], image_url=image_url)
    else:
        return render_template_string(HTML_TEMPLATE, title="处理失败", message=result["message"], image_url=None), 500


@app.route('/image', methods=['GET'])
def get_image():
    image_path = request.args.get('path')
    if not image_path or not os.path.exists(image_path):
        return "图像不存在", 404
    return send_file(image_path, mimetype='image/png')


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "DXF/DWG处理服务已启动",
        "endpoints": {
            "process_dxf": "GET /process_dxf?dxf_path=<path_to_dxf_or_dwg_file> (返回JSON结果)",
            "view_dxf": "GET /view_dxf?dxf_path=<path_to_dxf_or_dwg_file> (在网页中查看图像)",
            "image": "GET /image?path=<path_to_image> (获取图像文件)"
        },
        "notes": "支持DXF和DWG文件格式。DWG文件会自动转换为DXF格式后再处理。",
        "example": "GET /view_dxf?dxf_path=C:\\test\\example.dwg"
    })


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .message { padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 20px; }
        .error { background-color: #ffe6e6; border-left: 5px solid #ff0000; }
        .success { background-color: #e6ffe6; border-left: 5px solid #00cc00; }
        .info { background-color: #e6f3ff; border-left: 5px solid #0066cc; }
        .image-container { text-align: center; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .note { padding: 15px; background-color: #fff9e6; border-left: 5px solid #ffcc00; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        {% if message %}
        <div class="message {% if title == '处理失败' or title == '错误' or title == '转换失败' %}error{% elif title == '文件处理结果' %}success{% else %}info{% endif %}">
            <p>{{ message }}</p>
        </div>
        {% endif %}
        
        {% if image_url %}
        <div class="image-container">
            <h2>处理结果图像</h2>
            <img src="{{ image_url }}" alt="文件处理结果">
        </div>
        {% elif title != '错误' and title != '转换失败' %}
        <p>无法显示图像。</p>
        {% endif %}
        
        <div class="note">
            <h3>支持的文件格式：</h3>
            <ul>
                <li>DXF文件（直接处理）</li>
                <li>DWG文件（自动转换为DXF后处理）</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5301, debug=True)