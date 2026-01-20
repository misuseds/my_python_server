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
import os
from datetime import datetime
import win32gui  # 添加这个导入
import win32con  # 添加这个导入
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ocr_tools")

def get_window_rect_by_title(window_title_part: str):
    """根据窗口标题的一部分获取窗口矩形"""
    def enum_windows_callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title_part.lower() in window_title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                result.append(rect)
        return True

    result = []
    win32gui.EnumWindows(enum_windows_callback, result)
    return result[0] if result else None

def mask_window_area(image, window_title_part: str):
    """遮盖特定标题的窗口区域"""
    window_rect = get_window_rect_by_title(window_title_part)
    if window_rect is None:
        return image
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    left, top, right, bottom = window_rect
    
    # 遮盖窗口区域（使用黑色矩形）
    cv2.rectangle(img_cv, (left, top), (right, bottom), (0, 0, 0), -1)
    
    # 转换回PIL格式
    masked_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return masked_image

def save_ocr_image_with_boxes(image: Image.Image, results: List, image_path: str = None):
    """
    保存带识别框的图片
    """
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换PIL图像为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 在图片上绘制识别框
    for (bbox, text, confidence) in results:
        # 转换坐标为整数
        bbox_int = np.array(bbox, dtype=np.int32)
        # 绘制多边形框
        cv2.polylines(img_cv, [bbox_int], True, (0, 255, 0), 2)
        
        # 在框的左上角添加文字 - 确保坐标为整数类型
        x_min = int(min([int(point[0]) for point in bbox]))
        y_min = int(min([int(point[1]) for point in bbox]))
        cv2.putText(img_cv, str(text), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 生成文件名
    if image_path:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base_name = "screenshot"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_ocr_{timestamp}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存带识别框的图片
    cv2.imwrite(output_path, img_cv)
    
    return output_path
@mcp.tool()
def find_text_coordinates(text_to_find: str, image_path: str = None):
    """
    从指定图片或全屏截图中查找指定文字的中心点坐标，并保存带识别框的图片
    """
    try:
        # 如果没有提供图片路径，使用全屏截图
        if not image_path:
            screenshot = pyautogui.screenshot()
            # 遮盖VLM窗口区域
            image = mask_window_area(screenshot, "VLM牛马")
        else:
            # 加载指定的图片
            image = Image.open(image_path)
        
        # 初始化OCR读取器
        reader = easyocr.Reader(['ch_sim', 'en'])  # 可根据需要添加其他语言
        
        # 执行OCR识别
        results = reader.readtext(np.array(image))
        
        # 保存带识别框的图片
        saved_path = save_ocr_image_with_boxes(image, results, image_path)
      
        
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
                   
                    
                }
                
                return json.dumps(result, ensure_ascii=False, indent=2)
        return_text=False
        # 如果没有找到指定文字
        if return_text: 
            result = {
                "error": f"未找到文字: {text_to_find}",
                "found_texts": []  # 可以返回所有找到的文字作为参考
            }
                    # 添加所有找到的文字（可选，用于调试）
            for (bbox, text, confidence) in results:
                for keywords  in ["请勿", "禁止"] :
                    if keywords not in text.lower():  # 不区分大小写匹配

                        result["found_texts"].append(
                        text
                        
                        )
        else: 
            result = {
                 "error": f"未找到文字: {text_to_find}",
                
            }
    

        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {"error": f"查找文字坐标失败: {str(e)}"}
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def get_all_text(image_path: str = None):
    """
    从指定图片或全屏截图中获取所有识别到的文本，以字符串形式返回，并保存带识别框的图片
    """
    try:
        # 如果没有提供图片路径，使用全屏截图
        if not image_path:
            screenshot = pyautogui.screenshot()
            # 遮盖VLM窗口区域
            image = mask_window_area(screenshot, "VLM任务执行器")
        else:
            # 加载指定的图片
            image = Image.open(image_path)
        
        # 初始化OCR读取器
        reader = easyocr.Reader(['ch_sim', 'en'])  # 可根据需要添加其他语言
        
        # 执行OCR识别
        results = reader.readtext(np.array(image))
        
        # 保存带识别框的图片
        saved_path = save_ocr_image_with_boxes(image, results, image_path)
        print(f"带识别框的图片已保存到: {saved_path}")
        
        # 提取所有文本，按置信度排序
        all_text = []
        for (bbox, text, confidence) in results:
            all_text.append(text)
        
        # 将所有文本连接成一个字符串，每行一个文本
        full_text = '\n'.join(all_text)
        
        return full_text
    
    except Exception as e:
        return f"获取所有文本失败: {str(e)}"

if __name__ == '__main__':
    mcp.run()