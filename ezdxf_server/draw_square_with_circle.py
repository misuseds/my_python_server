import ezdxf
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

def draw_square_with_circle(length, diameter):
    """
    创建一个包含正方形和中心圆的DXF文档
    
    Args:
        length (float): 正方形边长
        diameter (float): 中心圆直径
    
    Returns:
        doc (ezdxf.document.Drawing): DXF文档对象
    """
    # 创建一个新的DXF文档
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # 计算正方形的坐标
    half_length = length / 2
    # 正方形四个角点坐标
    points = [
        (-half_length, -half_length),  # 左下角
        (half_length, -half_length),   # 右下角
        (half_length, half_length),    # 右上角
        (-half_length, half_length)    # 左上角
    ]
    
    # 绘制正方形（闭合多段线）
    msp.add_lwpolyline(points + [points[0]], close=True)
    
    # 绘制中心圆
    msp.add_circle(center=(0, 0), radius=diameter/2)
    
    return doc

def save_dxf_file(doc):
    """
    弹出文件保存对话框并保存DXF文件
    
    Args:
        doc (ezdxf.document.Drawing): 要保存的DXF文档
    """
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 确保对话框在最前面
    
    # 弹出保存文件对话框
    file_path = filedialog.asksaveasfilename(
        title="保存DXF文件",
        defaultextension=".dxf",
        filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
    )
    
    # 销毁根窗口
    root.destroy()
    
    # 如果用户选择了路径，则保存文件
    if file_path:
        try:
            doc.saveas(file_path)
            print(f"文件已保存至: {file_path}")
            return True
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
            return False
    else:
        print("操作已取消")
        return False

def get_user_input():
    """
    获取用户输入的参数
    """
    try:
        length = float(input("请输入正方形边长: "))
        diameter = float(input("请输入中心圆直径: "))
        
        # 验证输入值
        if length <= 0 or diameter <= 0:
            print("错误：长度和直径必须为正数")
            return None, None
            
        if diameter >= length:
            print("警告：圆的直径应小于正方形边长以确保圆在正方形内")
            confirm = input("是否继续？(y/n): ")
            if confirm.lower() != 'y':
                return None, None
                
        return length, diameter
    except ValueError:
        print("错误：请输入有效的数字")
        return None, None
    except KeyboardInterrupt:
        print("\n操作已取消")
        return None, None

def main():
    """
    主函数：获取用户输入并执行绘图和保存操作
    """
    print("开始创建正方形和中心圆的DXF文件...")
    
    # 获取用户输入
    length, diameter = get_user_input()
    
    # 检查输入是否有效
    if length is None or diameter is None:
        return
    
    try:
        # 创建图形
        print("正在创建图形...")
        doc = draw_square_with_circle(length, diameter)
        
        # 保存文件
        print("准备保存文件...")
        save_dxf_file(doc)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()