import tkinter as tk
from tkinter import filedialog
import pytesseract
from PIL import ImageGrab
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
import io
from PIL import Image
import pyautogui
import easyocr
import sys

def ocr_screen_tool():
    """
    弹窗选择图片并进行OCR识别，返回文字和坐标信息的JSON格式结果
    """
    try:
        # 创建tkinter根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 弹窗选择图片文件
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG文件", "*.png"),
                ("JPG文件", "*.jpg"),
                ("JPEG文件", "*.jpeg"),
                ("BMP文件", "*.bmp"),
                ("TIFF文件", "*.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        # 销毁根窗口
        root.destroy()
        
        # 如果用户取消选择，返回错误信息
        if not file_path:
            error_result = {"error": "用户取消了图片选择"}
            return json.dumps(error_result, ensure_ascii=False, indent=2)
        
        # 加载选择的图片
        image = Image.open(file_path)
        
        # 初始化OCR读取器
        reader = easyocr.Reader(['ch_sim', 'en'])  # 可根据需要添加其他语言
        
        # 执行OCR识别
        results = reader.readtext(np.array(image))
        
        # 格式化结果
        formatted_results = []
        for (bbox, text, confidence) in results:
            # 将NumPy类型转换为Python原生类型
            formatted_bbox = [
                [int(point[0]), int(point[1])] for point in bbox
            ]
            formatted_results.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": formatted_bbox
            })
        
        print("ocr:", formatted_results)
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)
    except Exception as e:
        error_result = {"error": f"OCR识别失败: {str(e)}"}
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def find_text_coordinates(text_to_find: str, image_path: str = None):
    """
    从指定图片或全屏截图中查找指定文字的中心点坐标
    """
    try:
        # 如果没有提供图片路径，使用全屏截图
        if not image_path:
            screenshot = pyautogui.screenshot()
            image = screenshot
        else:
            # 加载指定的图片
            image = Image.open(image_path)
        
        # 初始化OCR读取器
        reader = easyocr.Reader(['ch_sim', 'en'])  # 可根据需要添加其他语言
        
        # 执行OCR识别
        results = reader.readtext(np.array(image))
        
        # 遍历OCR结果查找指定文字
        for (bbox, text, confidence) in results:
            if text_to_find.lower() in text.lower():  # 不区分大小写匹配
                # 计算边界框的中心点坐标
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                center_x = int((min(x_coords) + max(x_coords)) / 2)
                center_y = int((min(y_coords) + max(y_coords)) / 2)
                
                result = {
                    "text": text,
                    "target_text": text_to_find,
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": [[int(point[0]), int(point[1])] for point in bbox],
                    "confidence": float(confidence)
                }
                
                return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 如果没有找到指定文字
        result = {
            "error": f"未找到文字: {text_to_find}",
            "found_texts": []  # 可以返回所有找到的文字作为参考
        }
        
        # 添加所有找到的文字（可选，用于调试）
        for (bbox, text, confidence) in results:
            result["found_texts"].append({
                "text": text,
                "confidence": float(confidence)
            })
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {"error": f"查找文字坐标失败: {str(e)}"}
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def get_all_text(image_path: str = None):
    """
    从指定图片或全屏截图中获取所有识别到的文本，以字符串形式返回
    """
    try:
        # 如果没有提供图片路径，使用全屏截图
        if not image_path:
            screenshot = pyautogui.screenshot()
            image = screenshot
        else:
            # 加载指定的图片
            image = Image.open(image_path)
        
        # 初始化OCR读取器
        reader = easyocr.Reader(['ch_sim', 'en'])  # 可根据需要添加其他语言
        
        # 执行OCR识别
        results = reader.readtext(np.array(image))
        
        # 提取所有文本，按置信度排序
        all_text = []
        for (bbox, text, confidence) in results:
            all_text.append(text)
        
        # 将所有文本连接成一个字符串，每行一个文本
        full_text = '\n'.join(all_text)
        
        return full_text
    
    except Exception as e:
        return f"获取所有文本失败: {str(e)}"

def main():
    """主函数，处理命令行参数以适配executor.py."""
    if len(sys.argv) < 2:
        # 当没有命令行参数时，直接运行ocr_screen_tool
        result = ocr_screen_tool()
        print(result)
        return
    
    tool_name = sys.argv[1]
    
    try:
        if tool_name == "ocr_screen":
            result = ocr_screen_tool()
        elif tool_name == "find_text_coordinates":
            if len(sys.argv) < 3:
                result = "错误: find_text_coordinates 需要指定要查找的文字"
            else:
                text_to_find = sys.argv[2]
                # 如果提供了图片路径参数
                image_path = sys.argv[3] if len(sys.argv) > 3 else None
                result = find_text_coordinates(text_to_find, image_path)
        elif tool_name == "get_all_text":
            # 如果提供了图片路径参数
            image_path = sys.argv[2] if len(sys.argv) > 2 else None
            result = get_all_text(image_path)
        else:
            result = f"错误: 未找到工具 '{tool_name}'"
    except Exception as e:
        result = f"执行工具时出错: {str(e)}"
    
    print(result)

if __name__ == "__main__":
    main()