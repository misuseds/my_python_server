import os
import sys
import re

def open_folder(folder_path):
    """
    打开指定文件夹 - 需要指定路径
    :param folder_path: 文件夹路径，必需参数
    """
    # 解析参数字符串，提取实际路径
    folder_path = parse_folder_path(folder_path)
    
    # 处理路径中的各种情况
    if folder_path:
        # 使用 os.path.normpath 来标准化路径
        folder_path = os.path.normpath(folder_path)
        folder_path = folder_path.strip().strip('"\'')
    
    # 检查文件夹是否存在且为目录
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            # 根据操作系统打开文件夹
            if sys.platform.startswith('win'):
                # Windows系统
                os.startfile(folder_path)
            elif sys.platform.startswith('darwin'):
                # macOS系统
                os.system(f'open "{folder_path}"')
            else:
                # Linux系统
                os.system(f'xdg-open "{folder_path}"')
            
            success_msg = f"已打开文件夹: {folder_path}"
            print(success_msg)
            return success_msg
        else:
            error_msg = f"路径存在但不是文件夹: {folder_path}"
            print(error_msg)
            return error_msg
    else:
        error_msg = f"文件夹不存在: {folder_path}"
        print(error_msg)
        return error_msg

def parse_folder_path(input_path):
    """
    从参数字符串中解析出实际路径
    持多种格式：
    - '[TOOL:open_folder,folder_path="E:\\blender"]' 
    - '[TOOL:open_folder(folder_path="E:/blender")]'
    """
    if not input_path:
        return input_path
    
    # 处理新格式：open_folder(folder_path="E:/blender")
    # 匹配括号内的 folder_path 参数
    pattern1 = r'open_folder\s*\(\s*folder_path\s*=\s*["\']?([^"\'\)]+)["\']?\s*\)'
    match1 = re.search(pattern1, input_path)
    
    if match1:
        extracted_path = match1.group(1)
        return extracted_path
    
    # 保持对旧格式的支持：open_folder,folder_path=
    pattern2 = r'folder_path[=\s]*"?([^"\]]+)"?'
    match2 = re.search(pattern2, input_path)
    
    if match2:
        extracted_path = match2.group(1)
        extracted_path = extracted_path.rstrip(']')
        return extracted_path
    else:
        # 如果没有匹配到，返回原始路径（可能是直接的路径）
        if '[' in input_path and ']' in input_path:
            result = input_path.strip('[]').split(',', 1)[-1]
            return result
        else:
            return input_path

def main():
    # 支持命令行参数，与现有工具系统兼容
    if len(sys.argv) > 2 and sys.argv[1] == "open_folder":
        custom_path = sys.argv[2] if len(sys.argv) > 2 else None
        if custom_path is None:
            print("错误: 必须提供文件夹路径参数")
            return
        open_folder(custom_path)

if __name__ == "__main__":
    main()