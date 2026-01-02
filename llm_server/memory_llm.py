import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
import subprocess
import re
import json
import sys
from pathlib import Path
from llm_class import VLMService  # 假设VLMService基于LLMService
import pyautogui
from PIL import Image
import base64
from io import BytesIO

# 定义全局变量
CURRENT_DIR = Path(__file__).parent
CONFIG_PATH = CURRENT_DIR / "tools_config.json"
KNOWLEDGE_FILE_PATH = CURRENT_DIR / "knowledge.txt"
CONFIG_PATH = CURRENT_DIR / "web_tools_config.json"
KNOWLEDGE_FILE_PATH = CURRENT_DIR / "web_knowledge.txt"
def execute_python_script(script_path, *args):
    """
    执行指定路径的Python脚本
    """
    # 获取项目根目录（从当前脚本位置向上一级）
    current_dir = Path(__file__).parent.parent  # 回到项目根目录
    script_full_path = current_dir / script_path
    
    if not script_full_path.exists():
        return f"错误: 脚本 '{script_path}' 不存在"
    
    if script_full_path.suffix != '.py':
        return f"错误: 文件必须是Python脚本 (.py文件)"
    
    try:
        # 构建命令：工具名称作为脚本的第一个参数
        cmd = [sys.executable, str(script_full_path)] + list(args)
       
        
        # 执行Python脚本，指定编码为UTF-8
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(current_dir),
            encoding='utf-8',      # 明确指定UTF-8编码
            errors='replace'       # 遇到编码错误时替换字符
        )
        print( "",result.stdout.strip())
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"脚本执行失败: {result.stderr.strip()}"
     
    except subprocess.TimeoutExpired:
        return f"脚本执行超时: {script_path}"
    except Exception as e:
        return f"执行脚本时出错: {str(e)}"


def list_available_tools():
    """
    从配置文件中列出所有可用工具
    """
    if not CONFIG_PATH.exists():
        return []
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('tools', [])
    except Exception as e:
        return []


def get_tool_by_name(tool_name):
    """
    根据工具名称获取工具信息
    """
    tools = list_available_tools()
    for tool in tools:
        if tool['name'] == tool_name:
            return tool
    return None


def execute_tool(tool_name, *args):
    """
    执行指定名称的工具
    """
    tool_info = get_tool_by_name(tool_name)
    if not tool_info:
        return f"错误: 未找到工具 '{tool_name}'"
    
    script_path = tool_info['path']
    result = execute_python_script(script_path, tool_name, *args)
   
    return result


def get_available_tools_info():
    """
    获取所有可用工具的信息
    """
    if not CONFIG_PATH.exists():
        return "错误: 工具配置文件不存在"
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('tools', [])
    except Exception as e:
        return []


def get_tools_description():
    """获取工具描述，用于提供给VLM"""
    tools = get_available_tools_info()
    if not tools or isinstance(tools, str):  # 检查是否返回错误
        return "当前没有可用工具"
    
    tools_desc = "可用工具列表:\n"
    for tool in tools:
        name = tool.get('name', '未知工具')
        desc = tool.get('description', '无描述')
        params = tool.get('parameters', [])
        
        if params:
            param_desc = ", ".join([f"{p['name']}({p['type']})" for p in params])
            tools_desc += f"- {name}: {desc} (参数: {param_desc})\n"
        else:
            tools_desc += f"- {name}: {desc} (无参数)\n"
    
    tools_desc += "\n使用格式: [TOOL:工具名称,参数1,参数2,...]\n"
    tools_desc += "\n任务完成标记: 任务完成后请输出 [TASK_COMPLETED] 来结束任务循环"
    return tools_desc


def image_to_base64(image_path):
    """将图像文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def is_task_completed(ai_response):
    """判断任务是否完成 - 使用特定结束标记"""
    # 使用特定的结束标记，而不是通用关键词
    completion_marker = "[TASK_COMPLETED]"
    
    if ai_response and completion_marker in ai_response:
        return True
    
    return False

def vision_task_loop(task_description, knowledge_file=None, memory_file=None, reset_first_iteration=True):
    """
    基于视觉的循环任务执行器
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if knowledge_file is None:
        knowledge_file = KNOWLEDGE_FILE_PATH  # 使用全局变量
    if memory_file is None:
        memory_file = os.path.join(current_dir, "memory.txt")
    
    # 创建LLM服务实例（模拟VLM）
    vlm_service = VLMService()
    
    # 读取固定知识
    system_prompt_parts = []
    
    # 添加可用工具信息到系统提示
    tools_description = get_tools_description()
    system_prompt_parts.append(f"可用工具信息:\n{tools_description}")
    
    # 添加固定知识
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_content = f.read()
            if knowledge_content.strip():
                system_prompt_parts.append(f"重要知识:\n{knowledge_content}")
    
    # 组合系统提示
    system_prompt = "\n".join(system_prompt_parts)
    
    iteration_count = 0
    max_iterations = 50  # 设置最大迭代次数，防止无限循环
    first_iteration = reset_first_iteration  # 使用参数来决定是否重置首次迭代标志
    
    # 记录之前的AI响应，用于检测重复行为
    previous_ai_response = ""
    previous_tool_result = ""
    
    while iteration_count < max_iterations:
        iteration_count += 1
        
        # 截取当前屏幕
        screenshot = pyautogui.screenshot()
        screenshot_path = os.path.join(current_dir, "current_screen.png")
        screenshot.save(screenshot_path)
        
        # 准备消息列表
        messages = []
        
        # 添加系统提示（包含工具信息）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 读取 memory 内容并加入到用户消息中
        memory_content = ""
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_content = f.read().strip()
        
        # 添加任务描述和当前截图信息 - 使用更清晰的格式
        user_message = f"当前任务: {task_description}\n"
        
        # 如果有 memory 内容，添加到用户消息中
        if memory_content:
            user_message += f"历史记忆:\n{memory_content}\n\n"
        
        user_message += "请分析当前屏幕截图，并继续执行任务。"
        
        # 构建包含图像的消息内容
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(screenshot_path)}"
            }
        }
        
        text_content = {
            "type": "text",
            "text": user_message
        }
        
        # 构建用户消息，包含文本和图像
        messages.append({
            "role": "user",
            "content": [text_content, image_content]
        })
        
      
        try:
            # 调用LLM服务（模拟VLM功能）
            result = vlm_service.create_with_image(messages)  # 不传递图像路径，因为已经在消息中包含
            ai_response = result['choices'][0]['message']['content']
            
          
            
            # 先显示AI分析
            yield f" {ai_response}"
            
            # 执行AI返回的工具指令
            tool_execution_result = process_tool_calls(ai_response, memory_file)
            
            # 更新历史记录
            previous_ai_response = ai_response
            previous_tool_result = tool_execution_result or ""
            
        
            if is_task_completed(ai_response or ""):
                    yield "任务已完成，退出循环"
                    break
            
            # 更新标志，表示不再是第一次迭代
            first_iteration = False

        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            yield error_msg
            break
    
    if iteration_count >= max_iterations:
        yield "达到最大迭代次数，停止任务执行"


def process_tool_calls(response_text, memory_file_path=None):
    """
    解析AI响应中的工具调用指令
    支持格式: [TOOL:工具名称,arg1,arg2,arg3...] 
    """
    # 修复正则表达式以正确捕获工具名称和所有参数
    tool_pattern = r'\[TOOL:([^\],\]]+)(?:,([^\]]*))?\]'
    matches = re.findall(tool_pattern, response_text)
    
    all_results = []
    if not matches: print ("未找到工具调用指令")
    for match in matches:
        tool_name = match[0]
        tool_args_str = match[1]  # 包含所有参数的字符串，可能为空

        # 验证工具是否存在
        tools = get_available_tools_info()
        if not tools or isinstance(tools, str):  # 检查是否返回错误
            all_results.append(f"工具 '{tool_name}' 执行失败: 无法获取工具列表")
            continue
            
        tool_exists = any(tool['name'] == tool_name for tool in tools)
        if not tool_exists:
            all_results.append(f"工具 '{tool_name}' 执行失败: 工具不存在")
            continue
        
        # 解析参数，处理带引号的参数值（如果存在参数）
        tool_args = []
        if tool_args_str:  # 如果有参数
            current_arg = ""
            inside_quotes = False
            quote_char = None
            
            i = 0
            while i < len(tool_args_str):
                char = tool_args_str[i]
                
                if char in ['"', "'"] and not inside_quotes:
                    # 开始引号
                    inside_quotes = True
                    quote_char = char
                elif char == quote_char and inside_quotes:
                    # 结束引号
                    inside_quotes = False
                    quote_char = None
                elif char == ',' and not inside_quotes:
                    # 参数分隔符，不在引号内
                    tool_args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
                i += 1
            
            # 添加最后一个参数
            if current_arg:
                tool_args.append(current_arg.strip())
       
        result = execute_tool(tool_name, *tool_args) if tool_args else execute_tool(tool_name)
        
        # 处理执行结果为None的情况
        if result is None:
            result = "工具执行结果为空"
        
        # 将所有工具的结果写入记忆文件，这样AI可以看到
        if memory_file_path:
            try:
                with open(memory_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"工具 '{tool_name}' 执行结果:\n{result}\n\n")
            except Exception as e:
                print(f"写入记忆文件失败: {e}")
        
        all_results.append(f"工具 '{tool_name}' 执行结果: {result}")
    
    return "\n".join(all_results) if all_results else None
def parse_history_content(content):
    """
    解析历史对话内容，转换为messages格式
    """
    messages = []
    lines = content.strip().split('\n')
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('用户:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_role = "user"
            current_content = [line[3:].strip()]
        elif line.startswith('AI:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_role = "assistant"
            current_content = [line[3:].strip()]
        elif line == "" and current_content:
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            current_role = None
            current_content = []
        elif current_role:
            current_content.append(line)
    
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": '\n'.join(current_content).strip()
        })
    
    return messages


class VLMTaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM任务执行器")
        # 修改窗口大小为较小尺寸并设置为置顶
        self.root.geometry("400x400")  # 调整为较小的尺寸
        self.root.attributes('-topmost', True)  # 设置窗口置topmost=True  # 设置窗口置顶
        
        # 任务执行标志
        self.is_executing = False 
        
        # 创建界面
        self.setup_ui()
        
        # 文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = KNOWLEDGE_FILE_PATH  # 使用全局变量
        self.memory_file = os.path.join(current_dir, "memory.txt")
        
        # 启动时加载记忆文件内容到显示区域
        self.load_memory_content()

    def setup_ui(self):
        # 任务描述输入区域
        task_frame = tk.Frame(self.root)
        task_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(task_frame, text="任务描述:").pack(anchor=tk.W)
        
        self.task_input = tk.Text(task_frame, height=3)
        self.task_input.pack(fill=tk.X, pady=5)
        
        # 控制按钮区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = tk.Button(
            control_frame,
            text="开始执行任务",
            command=self.start_task
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(
            control_frame,
            text="停止任务",
            command=self.stop_task,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 添加清除短期记忆按钮
        self.clear_memory_button = tk.Button(
            control_frame,
            text="清除短期记忆",
            command=self.clear_short_term_memory
        )
        self.clear_memory_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 添加常用命令按钮
        self.often_used_orders_button = tk.Button(
            control_frame,
            text="常用命令",
            command=self.show_often_used_orders_menu
        )
        self.often_used_orders_button.pack(side=tk.LEFT)
        
        # 聊天历史显示区域
        self.chat_history = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state='disabled',
            height=30
        )
        self.chat_history.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 任务状态标签
        self.status_label = tk.Label(self.root, text="状态: 等待任务开始", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_often_used_orders(self):
        """从文件中加载常用命令"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        orders_file = os.path.join(current_dir, "often_use_order.txt")
        
        if os.path.exists(orders_file):
            try:
                with open(orders_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 按行分割命令，过滤空行
                    orders = [line.strip() for line in content.split('\n') if line.strip()]
                    return orders
            except Exception as e:
                print(f"加载常用命令文件失败: {str(e)}")
                return []
        else:
            print(f"常用命令文件不存在: {orders_file}")
            return []

    def show_often_used_orders_menu(self):
        """显示常用命令菜单"""
        orders = self.load_often_used_orders()
        if not orders:
            messagebox.showinfo("提示", "没有找到常用命令")
            return
        
        # 创建弹出菜单
        menu = tk.Menu(self.root, tearoff=0)
        
        def insert_order(order_text):
            # 清空当前输入框内容并插入选中的命令
            self.task_input.delete(1.0, tk.END)
            self.task_input.insert(tk.END, order_text)
        
        # 为每个命令添加菜单项
        for i, order in enumerate(orders):
            menu.add_command(
                label=f"{i+1}. {order}",
                command=lambda o=order: insert_order(o)
            )
        
        # 显示菜单
        try:
            menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())
        except tk.TclError:
            # 如果无法在鼠标位置显示，就显示在窗口中心
            menu.post(self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50)
        finally:
            menu.grab_release()

    def load_memory_content(self):
        """启动时加载记忆文件内容到显示区域"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        # 启用文本框编辑
                        self.chat_history.config(state='normal')
                        # 清空当前内容
                        self.chat_history.delete(1.0, tk.END)
                        # 插入记忆文件内容
                        self.chat_history.insert(tk.END, content)
                        # 禁用编辑并滚动到底部
                        self.chat_history.config(state='disabled')
                        self.chat_history.see(tk.END)
            except Exception as e:
                print(f"加载记忆文件失败: {str(e)}")
        else:
            # 如果记忆文件不存在，清空显示区域
            self.chat_history.config(state='normal')
            self.chat_history.delete(1.0, tk.END)
            self.chat_history.config(state='disabled')

    def start_task(self):
        """开始执行任务"""
        task_description = self.task_input.get("1.0", tk.END).strip()
        if not task_description:
            messagebox.showwarning("警告", "请输入任务描述")
            return
        
        self.is_executing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("状态: 任务执行中...")
        
        # 在新线程中执行任务以避免界面冻结
        import threading
        task_thread = threading.Thread(
            target=self.run_task,
            args=(task_description,)
        )
        task_thread.daemon = True
        task_thread.start()

    def stop_task(self):
        """停止任务执行"""
        self.is_executing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("状态: 任务已停止")

    def run_task(self, task_description):
        """执行任务的主循环"""
        try:
            # 显示用户输入的任务
            self.display_message(f"用户: {task_description}")
            
            # 追加到记忆文件而不是覆盖
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"用户: {task_description}\n\n")
            
            # 执行任务循环，确保重置首次迭代标志
            for output in vision_task_loop(task_description, self.knowledge_file, self.memory_file, reset_first_iteration=True):
                if not self.is_executing:
                    self.display_message("系统: 任务已手动停止")
                    break
                
                self.display_message(f"AI: {output}")
                
                # 将输出追加到记忆文件
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"AI: {output}\n\n")

        except Exception as e:
            self.display_message(f"系统: 执行任务时出错: {str(e)}")
        finally:
            self.is_executing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.update_status("状态: 任务执行完成"))

    def display_message(self, message):
        """显示消息"""
        self.root.after(0, self._display_message, message)

    def _display_message(self, message):
        """在主线程中更新UI"""
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)


    def update_status(self, status_text):
        """更新状态栏"""
        self.status_label.config(text=status_text)

    def clear_short_term_memory(self):
        """手动清除短期记忆"""
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
                # 同时清空显示区域
                self.chat_history.config(state='normal')
                self.chat_history.delete(1.0, tk.END)
                self.chat_history.config(state='disabled')
                self.display_message( "短期记忆已手动清除")
            except Exception as e:
                messagebox.showerror("错误", f"清除短期记忆失败: {str(e)}")
        else:
            self.display_message("短期记忆文件不存在")

def main():
    root = tk.Tk()
    app = VLMTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()