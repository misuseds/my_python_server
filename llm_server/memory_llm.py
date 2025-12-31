import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
import subprocess
import re
import json
import sys
from pathlib import Path
from llm_server import VLMService  # 假设VLMService基于LLMService
import pyautogui
from PIL import Image
import base64
from io import BytesIO


def execute_tool(tool_name, *args):
    """
    执行指定名称的工具
    """
    current_dir = Path(__file__).parent
    executor_script = current_dir / "executor.py"
    
    if not executor_script.exists():
        return "错误: 执行器脚本不存在"
    
    cmd = [sys.executable, str(executor_script), tool_name] + list(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"工具执行失败: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"工具执行超时: {tool_name}"
    except Exception as e:
        return f"执行工具时出错: {str(e)}"


def get_available_tools_info():
    """
    获取所有可用工具的信息
    """
    current_dir = Path(__file__).parent
    config_path = current_dir / "tools_config.json"
    
    if not config_path.exists():
        return "错误: 工具配置文件不存在"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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
    return tools_desc


def image_to_base64(image_path):
    """将图像文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def is_task_completed(ai_response, tool_result):
    """判断任务是否完成"""
    completion_indicators = [
        "任务完成", "完成任务", "任务已结束", "已完成", "任务完成",
        "task completed", "finished", "done", "success", "成功"
    ]
    
    combined_text = f"{ai_response} {tool_result}".lower()
    return any(indicator in combined_text for indicator in completion_indicators)


def vision_task_loop(task_description, knowledge_file=None, memory_file=None, reset_first_iteration=True):
    """
    基于视觉的循环任务执行器
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if knowledge_file is None:
        knowledge_file = os.path.join(current_dir, "knowledge.txt")
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
        
        # 添加任务描述和当前截图信息 - 使用更清晰的格式
        if first_iteration:
            # 首次迭代时，AI只需要开始分析任务
            user_message = f"当前任务: {task_description}\n请分析当前屏幕截图，并开始执行任务。"
        else:
            # 非首次迭代时，询问任务完成情况
            user_message = f"当前任务: {task_description}\n请分析当前屏幕截图，判断任务完成情况，并按需执行相应操作。如果任务已经完成，请明确说明任务已完成。"
        
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
        
        print(f"用户消息: {user_message}, 截图保存在: {screenshot_path}")
        try:
            # 调用LLM服务（模拟VLM功能）
            result = vlm_service.create_with_image(messages)  # 不传递图像路径，因为已经在消息中包含
            ai_response = result['choices'][0]['message']['content']
            
            print(f"VLM响应: {ai_response}")
            
            # 执行AI返回的工具指令
            tool_execution_result = process_tool_calls(ai_response)
            print(f"工具执行结果: {tool_execution_result}")
            # 显示AI响应
            yield f"AI分析: {ai_response}"
            
            # 只在非首次迭代时检查任务完成状态
            if not first_iteration:
                # 检查任务是否完成
                if is_task_completed(ai_response, tool_execution_result or ""):
                    yield "任务已完成，退出循环"
                    break
            
            # 更新标志，表示不再是第一次迭代
            first_iteration = False
     
            # 如果没有工具执行结果，检查AI响应是否表明任务已完成
            if any(indicator in ai_response.lower() for indicator in 
                    ["任务完成", "完成任务", "已完成", "task completed", "finished", "done"]):
                yield "任务已完成，退出循环"
                # 取消自动删除短期记忆，改为手动删除
                break
        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            yield error_msg
            break
    
    if iteration_count >= max_iterations:
        yield "达到最大迭代次数，停止任务执行"
def process_tool_calls(response_text):
    """
    解析AI响应中的工具调用指令
    支持格式: [TOOL:工具名称,arg1,arg2,arg3...]
    """

    # 修复正则表达式以正确捕获工具名称和所有参数
    tool_pattern = r'\[TOOL:([^\],\]]+),([^\]]+)\]'
    matches = re.findall(tool_pattern, response_text)
    

    all_results = []
    
    if matches:
        for match in matches:
            tool_name = match[0]
            tool_args_str = match[1]  # 包含所有参数的字符串
            
            print(f"DEBUG: 解析到工具名称: '{tool_name}', 参数字符串: '{tool_args_str}'")
            
            # 验证工具是否存在
            tools = get_available_tools_info()
            if not tools or isinstance(tools, str):  # 检查是否返回错误
                all_results.append(f"工具 '{tool_name}' 执行失败: 无法获取工具列表")
                continue
                
            tool_exists = any(tool['name'] == tool_name for tool in tools)
            if not tool_exists:
                all_results.append(f"工具 '{tool_name}' 执行失败: 工具不存在")
                continue
            
            # 正确解析参数，处理带引号的参数值
            tool_args = []
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
            
        
            result = execute_tool(tool_name, *tool_args)
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
        self.root.geometry("400x600")  # 调整为较小的尺寸
        self.root.attributes('-topmost', True)  # 设置窗口置顶
        
        # 任务执行标志
        self.is_executing = False 
        
        # 创建界面
        self.setup_ui()
        
        # 文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = os.path.join(current_dir, "knowledge.txt")
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
        self.clear_memory_button.pack(side=tk.LEFT)
        
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
            self.display_message("用户", task_description)
            
            # 追加到记忆文件而不是覆盖
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"用户: {task_description}\n\n")
            
            # 执行任务循环，确保重置首次迭代标志
            for output in vision_task_loop(task_description, self.knowledge_file, self.memory_file, reset_first_iteration=True):
                if not self.is_executing:
                    break
                
                self.display_message("系统", output)
                
                # 将输出追加到记忆文件
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"系统: {output}\n\n")
        
        except Exception as e:
            self.display_message("系统", f"执行任务时出错: {str(e)}")
        finally:
            self.is_executing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.update_status("状态: 任务执行完成"))    
    def display_message(self, sender, message):
        """显示消息"""
        self.root.after(0, self._display_message, sender, message)

    def _display_message(self, sender, message):
        """在主线程中更新UI"""
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n\n")
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
                self.display_message("系统", "短期记忆已手动清除")
            except Exception as e:
                messagebox.showerror("错误", f"清除短期记忆失败: {str(e)}")
        else:
            self.display_message("系统", "短期记忆文件不存在")
def main():
    root = tk.Tk()
    app = VLMTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()