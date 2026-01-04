import os
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import json

def select_file_via_dialog():
    """通过文件选择对话框选择要打包的Python文件"""
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 置于顶层
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择要打包的Python文件",
        filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
        initialdir=os.getcwd()
    )
    
    # 销毁根窗口
    root.destroy()
    
    return file_path

def build_exe(script_path):
    """打包指定的Python文件"""
    # 获取文件名和目录
    script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)
    
    # 保存当前工作目录
    original_dir = os.getcwd()
    
    # 定义输出文件夹（使用绝对路径）
    output_folder = os.path.abspath(os.path.join(original_dir, "exe_output"))
    
    # 获取不带扩展名的文件名作为exe名称
    exe_name = os.path.splitext(script_name)[0]
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建文件夹: {output_folder}")
    
    # 构建exe文件
    print(f"正在打包 {script_path}...")
    try:
        # 使用PyInstaller打包，使用绝对路径调用pyinstaller.exe
        pyinstaller_path = r"E:\code\my_python_server\micromambavenv\Scripts\pyinstaller.exe"
        
        # 检查 pyinstaller.exe 是否存在
        if not os.path.exists(pyinstaller_path):
            print(f"错误: 找不到 pyinstaller 可执行文件: {pyinstaller_path}")
            return False
        
        # 检查当前脚本所在目录是否有favicon.ico文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "favicon.ico")
        
        if os.path.exists(icon_path):
            # 验证图标文件格式
            print(f"使用图标文件: {icon_path}")
            print(f"图标文件大小: {os.path.getsize(icon_path)} 字节")
            
            # 检查图标文件头是否正确 (ICO文件应该以 0x00 0x00 0x01 0x00 开头)
            try:
                with open(icon_path, 'rb') as f:
                    header = f.read(4)
                    expected_header = b'\x00\x00\x01\x00'  # ICO文件头
                    if header == expected_header:
                        print("图标文件头验证通过")
                    else:
                        print(f"警告: 图标文件头不正确，期望 {expected_header.hex()}，实际 {header.hex()}")
            except Exception as e:
                print(f"验证图标文件头时出错: {e}")
                
        else:
            print(f"警告: 未找到图标文件 {icon_path}，EXE将使用默认图标")
        
        # 构建命令行参数
        cmd = [
            pyinstaller_path,
            "--noconfirm",            # 不询问确认
            "--clean",                # 清理缓存
            "--onefile",              # 打包成单个文件
            "--name", exe_name,       # exe文件名使用原文件名
            "--distpath", output_folder,  # 指定输出目录
            "--hidden-import=llm_class",  # 隐式导入llm_class模块
            "--hidden-import=tkinter",    # 隐式导入tkinter模块
            "--hidden-import=tkinter.scrolledtext",  # 隐式导入scrolledtext模块
            "--hidden-import=tkinter.messagebox",     # 隐式导入messagebox模块
            "--hidden-import=tkinter.ttk",            # 隐式导入ttk模块
            "--hidden-import=pyautogui",  # 隐式导入pyautogui模块
            "--hidden-import=base64",     # 隐式导入base64模块
            "--hidden-import=subprocess", # 隐式导入subprocess模块
            "--hidden-import=json",       # 隐式导入json模块
            "--hidden-import=re",         # 隐式导入re模块
            "--hidden-import=sys",        # 隐式导入sys模块
            "--hidden-import=pathlib",    # 隐式导入pathlib模块
            "--add-data", f"{script_dir}{os.pathsep}config",  # 添加config目录
            "--add-data", f"{os.path.join(current_dir, '.env')}{os.pathsep}.",  # 添加.env文件
            "--paths", script_dir,        # 添加脚本目录到Python路径
        ]
        
        # 检查当前脚本所在目录是否有favicon.ico文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "favicon.ico")
        
        if os.path.exists(icon_path):
            # 验证图标文件是否有效后再添加
            with open(icon_path, 'rb') as f:
                header = f.read(4)
                if header == b'\x00\x00\x01\x00':  # ICO文件头
                    cmd.extend([f"--icon={icon_path}"])  # 使用 --icon 参数设置EXE图标
                    print(f"使用图标文件: {icon_path}")
                else:
                    print(f"图标文件格式无效，跳过使用图标: {icon_path}")
        else:
            print(f"警告: 未找到图标文件 {icon_path}，EXE将使用默认图标")
        
        # 添加主脚本路径
        cmd.append(script_path)
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 输出详细信息以便调试
        if result.stdout:
            print("PyInstaller 输出:")
            print(result.stdout)
        if result.stderr:
            print("PyInstaller 错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("打包成功!")
            exe_path = os.path.join(output_folder, f"{exe_name}.exe")
            print(f"EXE文件应保存到: {exe_path}")
            
            # 检查文件是否真的存在
            if os.path.exists(exe_path):
                print("EXE文件确认存在!")
            else:
                print("警告: EXE文件未在预期位置找到")
                # 检查默认输出位置
                default_dist_path = os.path.join(original_dir, "dist", f"{exe_name}.exe")
                if os.path.exists(default_dist_path):
                    print(f"EXE文件在默认位置找到: {default_dist_path}")
                    # 移动到正确的输出文件夹
                    shutil.move(default_dist_path, exe_path)
                    print(f"已移动EXE文件到: {exe_path}")
            
            return True
        else:
            print("打包失败:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"打包过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件夹
        for temp_dir in ["build", "dist", "__pycache__"]:  # 添加__pycache__清理
            temp_path = os.path.join(original_dir, temp_dir)
            if os.path.exists(temp_path):
                try:
                    shutil.rmtree(temp_path)
                    print(f"已清理临时文件夹: {temp_dir}")
                except Exception as e:
                    print(f"清理临时文件夹 {temp_dir} 失败: {e}")

def main():
    print("Python转EXE打包工具")
    print("=" * 30)
    
    # 通过对话框选择文件
    selected_file = select_file_via_dialog()
    
    if not selected_file:
        print("未选择文件，程序退出")
        return
    
    # 检查是否为Python文件
    if not selected_file.endswith('.py'):
        print("错误: 请选择Python文件(.py)")
        return
    
    # 检查文件是否存在
    if not os.path.exists(selected_file):
        print(f"错误: 文件 {selected_file} 不存在")
        return
    
    # 检查是否安装了PyInstaller
    try:
        pyinstaller_path = r"E:\code\my_python_server\micromambavenv\Scripts\pyinstaller.exe"
        
        # 检查 pyinstaller.exe 是否存在
        if not os.path.exists(pyinstaller_path):
            raise Exception("PyInstaller executable not found")
            
        cmd = [pyinstaller_path, "--version"]
        print(f"检查PyInstaller: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception("PyInstaller not found")
        print(f"找到PyInstaller版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print("错误: 未检测到PyInstaller，请先安装PyInstaller")
        print("可以通过以下命令安装: pip install pyinstaller")
        print(f"错误详情: {e}")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("缺少依赖", "未检测到PyInstaller，请先安装PyInstaller\n\n可以通过以下命令安装:\npip install pyinstaller")
        root.destroy()
        return
    
    # 检查是否存在llm_class.py文件
    script_dir = os.path.dirname(selected_file)
    llm_class_path = os.path.join(script_dir, "llm_class.py")
    if not os.path.exists(llm_class_path):
        print(f"警告: 未找到 llm_class.py 文件: {llm_class_path}")
        print("如果您的项目依赖此模块，打包可能失败")
    
    # 打包选定的文件
    print(f"\n开始打包: {selected_file}")
    success = build_exe(selected_file)
    
    # 显示结果
    if success:
        print("\n打包完成!")
        # 显示成功消息框
        root = tk.Tk()
        root.withdraw()
        output_folder = os.path.abspath("exe_output")
        exe_name = os.path.splitext(os.path.basename(selected_file))[0]
        messagebox.showinfo("打包完成", f"EXE文件已成功生成!\n\n保存位置:\n{output_folder}\\{exe_name}.exe")
        root.destroy()
    else:
        print("\n打包失败!")
        # 显示错误消息框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("打包失败", "打包过程中出现错误，请查看控制台输出")
        root.destroy()

if __name__ == "__main__":
    main()