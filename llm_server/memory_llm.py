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

CONFIG_PATH = CURRENT_DIR /"config"/ "web_tools_config.json"
KNOWLEDGE_FILE_PATH = CURRENT_DIR /"config"/ "web_knowledge.txt"
OFTEN_USE_ORDER_PATH = CURRENT_DIR /"config"/ "web_often_use_order.txt" 
WORKFLOW_PATH = CURRENT_DIR /"config"/ "web_workflow.txt"  # 新增：工作流程文件路径

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
    tools_desc += "\n任务完成标记: 每完成一个步骤后，必须在响应末尾输出 [TASK_COMPLETED] 来标记步骤完成，否则系统将不会继续下一个步骤"
    tools_desc += "\n工作流程完成标记: 当所有工作流程完成时，请输出 [TOTAL_TASK_COMPLETED]"
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

def is_workflow_completed(ai_response):
    """判断工作流程是否完成 - 使用特殊标记"""
    completion_marker = "[TOTAL_TASK_COMPLETED]"
    
    if ai_response and completion_marker in ai_response:
        return True
    
    return False

def vision_task_loop(task_description, knowledge_file=None, memory_file=None, workflow_state=None, reset_first_iteration=True, gui_callback=None):
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
    
    # 添加工作流程状态信息 - 只显示当前任务及之前已完成/待确定的任务
    if workflow_state:
        workflow_status = "当前工作流程状态:\n"
        current_task_index = -1
        
        # 找到当前任务在工作流程中的位置
        for i, (step, completed) in enumerate(workflow_state):
            if task_description in step:
                current_task_index = i
                break
        
        # 如果没找到完全匹配的，尝试部分匹配
        if current_task_index == -1:
            for i, (step, completed) in enumerate(workflow_state):
                if task_description.strip() in step or step in task_description:
                    current_task_index = i
                    break
        
        # 只显示当前任务及之前已完成或待确定的任务
        for i, (step, completed) in enumerate(workflow_state):
            if i <= current_task_index:  # 只显示当前及之前的任务
                if completed == True:
                    status = "已完成"
                elif completed == "pending_verification":
                    status = "待确定"
                else:
                    status = "待完成"
                workflow_status += f"任务{i+1}: {step} - {status}\n"
            else:
                # 对于未执行的任务，只显示任务号和名称，不显示状态
                workflow_status += f"任务{i+1}: {step}\n"
        system_prompt_parts.append(workflow_status)
    
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
        
        # 添加系统提示（包含工具信息和工作流程状态）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 读取 memory 内容并加入到用户消息中 - 只包含已执行任务的历史
        memory_content = ""
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 分离工作流程定义和执行历史
            lines = content.split('\n')
            workflow_part = []
            history_part = []
            in_history = False
            
            for line in lines:
                if line == "执行历史:":
                    in_history = True
                    history_part.append(line)
                    continue
                elif line.startswith("工作流程定义:"):
                    workflow_part.append(line)
                    continue
                elif line.startswith("任务") and " - " in line and not in_history:
                    # 只包含已完成或待确定的任务
                    if "已完成" in line or "待确定" in line or "执行完成" in line or "确认完成" in line:
                        workflow_part.append(line)
                    continue
                elif in_history:
                    history_part.append(line)
                else:
                    workflow_part.append(line)
            
            memory_content = '\n'.join(workflow_part + history_part)
        
        # 添加任务描述和当前截图信息 - 使用更清晰的格式
        user_message = f"当前任务: {task_description}\n"
        
        # 如果有 memory 内容，添加到用户消息中
        if memory_content:
            user_message += f"历史记忆:\n{memory_content}\n\n"
        
        # 只在非第一轮迭代时检查之前的完成标记
        if iteration_count > 1:
            if previous_ai_response and not (is_task_completed(previous_ai_response) or is_workflow_completed(previous_ai_response)):
                user_message += "请注意：上一轮AI响应中没有检测到任务完成标记 [TASK_COMPLETED]，任务尚未完成，请继续执行任务。\n"
            elif previous_ai_response and (is_task_completed(previous_ai_response) or is_workflow_completed(previous_ai_response)):
                user_message += "任务已完成，请输出 [TOTAL_TASK_COMPLETED] 标记整个工作流程完成，或者继续执行剩余任务。\n"
        else:
            # 第一轮迭代，只需提示AI开始执行任务
            user_message += "请分析当前屏幕截图，并开始执行任务。注意：每完成一个步骤后，必须在响应末尾输出 [TASK_COMPLETED] 来标记步骤完成，否则系统将不会继续下一个步骤。"
        
        print("user_message[",user_message,"]")
        
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
            
            # 通过回调函数在GUI中显示LLM服务返回结果
            if gui_callback:
                gui_callback(f"LLM服务返回: 【{ai_response}】")
            
            # 先显示AI分析
            yield f" {ai_response}"
            
            # 执行AI返回的工具指令 - 传递GUI回调
            tool_execution_result = process_tool_calls(ai_response, memory_file, gui_callback)
            
            # 检查是否包含完成标记
            if is_task_completed(ai_response or "") or is_workflow_completed(ai_response or ""):
                yield "任务已完成，退出循环"
                break  # 立即退出循环，不再继续
            
            # 更新历史记录
            previous_ai_response = ai_response
            previous_tool_result = tool_execution_result or ""
            
            # 更新标志，表示不再是第一次迭代
            first_iteration = False

        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            yield error_msg
            break
    
    if iteration_count >= max_iterations:
        yield "达到最大迭代次数，停止任务执行"

def process_tool_calls(response_text, memory_file_path=None, gui_callback=None):
    """
    解析AI响应中的工具调用指令
    支持格式: [TOOL:工具名称,arg1,arg2,arg3...] 
    添加gui_callback参数用于GUI显示
    """
    # 检测任务完成标记
    task_completed = is_task_completed(response_text)
    workflow_completed = is_workflow_completed(response_text)
    
    if task_completed and memory_file_path:
        with open(memory_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n检测到任务完成标记: [TASK_COMPLETED]\n")
    
    # 检测工作流程完成标记
    if workflow_completed and memory_file_path:
        with open(memory_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n检测到工作流程完成标记: [TOTAL_TASK_COMPLETED]\n")
    
    # 在GUI中显示完成标记检测
    if gui_callback:
        if task_completed:
            gui_callback("检测到任务完成标记: [TASK_COMPLETED]")
        if workflow_completed:
            gui_callback("检测到工作流程完成标记: [TOTAL_TASK_COMPLETED]")
    
    # 修复正则表达式以正确捕获工具名称和所有参数
    tool_pattern = r'\[TOOL:([^\],\]]+)(?:,([^\]]*))?\]'
    matches = re.findall(tool_pattern, response_text)
    
    all_results = []
    if not matches: 
        # 只在响应中包含工具调用格式但未找到匹配时才输出，而不是所有情况
        if '[TOOL:' in response_text:
            print("未找到工具调用指令")
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
        
        # 通过回调函数在GUI中显示工具执行结果
        if gui_callback:
            gui_callback(f"工具 '{tool_name}' 执行结果: {result}")
        
        all_results.append(f"工具 '{tool_name}' 执行结果: {result}")
    
    # 返回工具执行结果和完成状态
    return {
        "results": "\n".join(all_results) if all_results else None,
        "task_completed": task_completed,
        "workflow_completed": workflow_completed
    }


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
        # 修改窗口大小 - 调整为更窄的尺寸
        self.root.geometry("400x300")
        self.root.attributes('-topmost', True)  # 设置窗口置顶
        
        # 任务执行标志
        self.is_executing = False 
        self.workflow_state = []  # 工作流程状态
        self.current_executing_step = -1  # 当前正在执行的步骤索引
        self.current_page_index = 0  # 当前显示的页面索引
        
        # 创建界面
        self.setup_ui()
        
        # 文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = KNOWLEDGE_FILE_PATH  # 使用全局变量
        self.memory_file = os.path.join(current_dir, "memory.txt")
        
        # 首先加载工作流程（从工作流文件和记忆文件获取状态）
        self.load_workflow_content()
        # 然后加载记忆文件内容到显示区域
        self.load_memory_content()
        
    def setup_ui(self):
        # 主容器
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 任务标题
        self.title_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 14, "bold"),
            anchor="w"
        )
        self.title_label.pack(fill=tk.X, pady=(0, 5))

        # 任务描述
        self.desc_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 11),
            wraplength=350,
            justify=tk.LEFT,
            anchor="nw"
        )
        self.desc_label.pack(fill=tk.X, pady=(0, 10))

        # 执行结果区域 - 减小高度
        result_frame = tk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(result_frame, text="执行结果:", font=("Arial", 10, "bold")).pack(anchor="w")

        # 创建滚动文本框显示执行结果 - 减小高度
        text_frame = tk.Frame(result_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=3  # 减小高度
        )
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 控制按钮区域 - 第一行：上一页、页码、下一页
        control_frame1 = tk.Frame(self.root)
        control_frame1.pack(fill=tk.X, padx=10, pady=5)

        self.prev_button = tk.Button(
            control_frame1,
            text="上一页",
            command=self.prev_page,
            state=tk.DISABLED
        )
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))

        self.page_label = tk.Label(
            control_frame1,
            text="第 0/0 页",
            width=15
        )
        self.page_label.pack(side=tk.LEFT, padx=(0, 5))

        self.next_button = tk.Button(
            control_frame1,
            text="下一页",
            command=self.next_page,
            state=tk.DISABLED
        )
        self.next_button.pack(side=tk.LEFT, padx=(0, 5))

        # 控制按钮区域 - 第二行：执行当前任务、停止、清除记忆
        control_frame2 = tk.Frame(self.root)
        control_frame2.pack(fill=tk.X, padx=10, pady=5)

        self.run_all_button = tk.Button(
            control_frame2,
            text="执行当前任务",
            command=self.run_current_task
        )
        self.run_all_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = tk.Button(
            control_frame2,
            text="停止",
            command=self.stop_all_tasks,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_memory_button = tk.Button(
            control_frame2,
            text="清除记忆",
            command=self.clear_short_term_memory
        )
        self.clear_memory_button.pack(side=tk.LEFT, padx=(0, 5))

        # 任务状态标签
        self.status_label = tk.Label(self.root, text="状态: 等待任务开始", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_workflow_content(self):
        """加载并显示工作流程内容"""
        # 首先尝试从记忆文件中获取步骤完成状态
        saved_state = self.get_workflow_state_from_memory()
        
        if WORKFLOW_PATH.exists():
            try:
                with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 分割工作流程为独立步骤
                steps = [step.strip() for step in content.split('\n') if step.strip()]
                
                # 初始化工作流程状态，优先使用记忆文件中的状态
                self.workflow_state = []
                for i, step in enumerate(steps):
                    completed = False
                    # 检查记忆文件中是否有该步骤的完成状态
                    if i < len(saved_state):
                        _, saved_completed = saved_state[i]
                        completed = saved_completed
                    self.workflow_state.append((step, completed))
                
                # 更新页面导航
                self.update_page_navigation()
                
                # 显示第一页
                if self.workflow_state:
                    self.update_task_display()
                
            except Exception as e:
                print(f"加载工作流程失败: {str(e)}")
        else:
            print(f"工作流程文件不存在: {WORKFLOW_PATH}")

    def update_task_display(self):
        """更新当前任务显示"""
        if not self.workflow_state or self.current_page_index >= len(self.workflow_state):
            return
            
        step, completed = self.workflow_state[self.current_page_index]
        
        # 更新标题
        if completed == True:
            status_text = "已完成"
            status_color = "green"
        elif completed == "pending_verification":
            status_text = "待确定"
            status_color = "orange"
        else:
            status_text = "待完成"
            status_color = "red"
            
        self.title_label.config(
            text=f"任务 {self.current_page_index + 1}/{len(self.workflow_state)} - {status_text}",
            fg=status_color
        )
        
        # 更新描述
        self.desc_label.config(text=step)

    def update_page_navigation(self):
        """更新页面导航按钮状态"""
        total_pages = len(self.workflow_state)
        
        # 更新按钮状态
        self.prev_button.config(state=tk.NORMAL if self.current_page_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_page_index < total_pages - 1 else tk.DISABLED)
        
        # 更新页面标签
        self.page_label.config(text=f"第 {self.current_page_index + 1}/{total_pages} 页")

    def update_page_label(self):
        """更新页面标签"""
        total_pages = len(self.workflow_state)
        self.page_label.config(text=f"第 {self.current_page_index + 1}/{total_pages} 页")

    def prev_page(self):
        """上一页"""
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.update_task_display()
            self.update_page_navigation()
            
            # 清空执行结果区域
            self.clear_result_display()

    def next_page(self):
        """下一页"""
        if self.current_page_index < len(self.workflow_state) - 1:
            self.current_page_index += 1
            self.update_task_display()
            self.update_page_navigation()
            
            # 清空执行结果区域
            self.clear_result_display()

    def clear_result_display(self):
        """清空执行结果区域"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

    def run_current_task(self):
        """执行当前任务"""
        if self.is_executing:
            messagebox.showwarning("警告", "任务正在执行中，请等待完成")
            return
        
        if not self.workflow_state:
            messagebox.showwarning("警告", "没有可执行的任务")
            return
        
        current_task_index = self.current_page_index
        if current_task_index >= len(self.workflow_state):
            messagebox.showwarning("警告", "当前页码超出任务范围")
            return
            
        task_step, completed = self.workflow_state[current_task_index]
        
        if completed == True:
            messagebox.showinfo("提示", f"任务 {current_task_index + 1} 已完成，无需再次执行")
            return
        
        self.is_executing = True
        self.run_all_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status(f"状态: 正在执行任务 {current_task_index + 1}: {task_step}")
        
        # 在新线程中执行任务以避免界面冻结
        import threading
        task_thread = threading.Thread(
            target=self.execute_single_task,
            args=(current_task_index,)
        )
        task_thread.daemon = True
        task_thread.start()

    def execute_single_task(self, task_index):
        """执行单个任务"""
        try:
            task_step, completed = self.workflow_state[task_index]
            
            if completed == True:
                self.display_message(f"任务 {task_index + 1} 已完成")
                return
            
            # 定义GUI回调函数
            def gui_callback(msg):
                self.display_message(msg)
            
            # 执行任务 - 传递GUI回调函数
            task_output = ""
            completed_flag_found = False
            for output in vision_task_loop(
                task_step, 
                self.knowledge_file, 
                self.memory_file, 
                self.workflow_state, 
                reset_first_iteration=True,
                gui_callback=gui_callback
            ):
                if not self.is_executing:
                    self.display_message("系统: 任务已手动停止")
                    return
                
                # 检查输出是否表示任务完成
                if "任务已完成，退出循环" in output:
                    completed_flag_found = True
                    # 自动标记为待确定状态
                    self.root.after(0, lambda idx=task_index: self.mark_step_as_pending_verification(idx))
                    break
                else:
                    # 检查是否包含完成标记
                    if is_task_completed(output) or is_workflow_completed(output):
                        self.display_message(f"任务{task_index + 1}执行结果: {output}")
                        # 自动标记为待确定状态
                        self.root.after(0, lambda idx=task_index: self.mark_step_as_pending_verification(idx))
                        break
                    else:
                        self.display_message(f"任务{task_index + 1}执行结果: {output}")
                        
                        # 将输出追加到记忆文件
                        with open(self.memory_file, 'a', encoding='utf-8') as f:
                            f.write(f"任务{task_index + 1}执行结果: {output}\n")
                        
                        task_output += output + "\n"
            
            # 如果循环结束但没有找到完成标记，说明达到了最大迭代次数
            if not completed_flag_found:
                self.display_message(f"任务{task_index + 1}未能在最大迭代次数内完成，未找到[TASK_COMPLETED]标记")
                return
            
            # 检查是否完成整个工作流程
            if "[TOTAL_TASK_COMPLETED]" in task_output:
                self.display_message("工作流程已全部完成")
                
        except Exception as e:
            self.display_message(f"系统: 执行任务时出错: {str(e)}")
        finally:
            self.is_executing = False
            self.root.after(0, lambda: self.run_all_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.update_status("状态: 任务执行完成"))

    def mark_step_as_pending_verification(self, index):
        """标记步骤为待确定状态并自动翻页"""
        if 0 <= index < len(self.workflow_state):
            # 将任务状态改为待确定而不是已完成
            self.workflow_state[index] = (self.workflow_state[index][0], "pending_verification")
            
            # 保存状态
            self.save_workflow_state()
            
            # 更新当前显示（如果当前页是完成的页）
            if index == self.current_page_index:
                self.update_task_display()
            
            # 在记忆文件中记录步骤执行完成，等待确认
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"任务{index+1} 执行完成，等待确认: {self.workflow_state[index][0]}\n")
            
            # 自动翻页到下一页（如果存在）
            if index == self.current_page_index and index < len(self.workflow_state) - 1:
                self.current_page_index += 1
                self.update_task_display()
                self.update_page_navigation()

    def confirm_task_completed(self, task_index):
        """确认任务真正完成"""
        if 0 <= task_index < len(self.workflow_state):
            task_desc, status = self.workflow_state[task_index]
            if status == "pending_verification":
                self.workflow_state[task_index] = (task_desc, True)
                self.save_workflow_state()
                
                # 更新显示（如果当前页是确认的页）
                if task_index == self.current_page_index:
                    self.update_task_display()
                
                # 在记忆文件中记录步骤确认完成
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"任务{task_index+1} 确认完成: {task_desc}\n")

    def stop_all_tasks(self):
        """停止所有任务执行"""
        self.is_executing = False
        self.run_all_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("状态: 任务已停止")

    def get_workflow_state_from_memory(self):
        """从记忆文件中提取工作流程状态"""
        saved_state = []
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                in_workflow_section = False
                
                for line in lines:
                    line = line.strip()
                    
                    if line == "工作流程定义:":
                        in_workflow_section = True
                        continue
                    elif line == "执行历史:":
                        in_workflow_section = False
                        continue
                    elif in_workflow_section and line.startswith("任务") and " - " in line:
                        # 解析格式如 "任务1: 打开模之屋 - 已完成"
                        try:
                            parts = line.split(" - ")
                            if len(parts) >= 2:
                                step_info = parts[0]  # "任务1: 打开模之屋"
                                status = parts[1]     # "已完成" 或 "待完成" 或 "待确定"
                                
                                # 提取任务号
                                step_match = re.search(r'任务(\d+):', step_info)
                                if step_match:
                                    step_num = int(step_match.group(1))
                                    if status == "已完成":
                                        completed = True
                                    elif status == "待确定":
                                        completed = "pending_verification"
                                    else:
                                        completed = False
                                    
                                    # 确保列表长度足够
                                    while len(saved_state) < step_num:
                                        saved_state.append((None, False))
                                    
                                    # 更新对应位置的完成状态
                                    step_desc = step_info.split(":", 1)[1].strip()
                                    saved_state[step_num - 1] = (step_desc, completed)
                        except Exception as e:
                            print(f"解析记忆中的任务行出错: {line}, 错误: {str(e)}")
                            continue
            except Exception as e:
                print(f"从记忆文件读取工作流程状态失败: {str(e)}")
        
        return saved_state

    def load_memory_content(self):
        """启动时加载记忆文件内容到显示区域"""
        if os.path.exists(self.memory_file):
            try:                 
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if content.strip():
                    completed_count = len([s for s, c in self.workflow_state if c == True])
                    pending_count = len([s for s, c in self.workflow_state if c == "pending_verification"])
                    self.display_message(f"加载历史记忆: {completed_count}个任务已完成, {pending_count}个任务待确定")
                    
            except Exception as e:
                print(f"加载记忆文件失败: {str(e)}")

    def display_message(self, message):
        """显示消息到状态栏"""
        self.status_label.config(text=f"状态: {message[:50]}...")  # 限制显示长度
        
        # 同时在结果区域显示
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)

    def update_status(self, status_text):
        """更新状态栏"""
        self.status_label.config(text=status_text)

    def clear_short_term_memory(self):
        """手动清除短期记忆"""
        if os.path.exists(self.memory_file):
            try:
                # 保留工作流程定义，只清除执行历史
                with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
                    workflow_content = f.read()
                
                # 重写memory文件，只保留工作流程定义
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    f.write("工作流程定义:\n")
                    steps = [step.strip() for step in workflow_content.split('\n') if step.strip()]
                    for i, step in enumerate(steps):
                        f.write(f"任务{i+1}: {step} - 待完成\n")
                    f.write("\n执行历史:\n")
                
                # 重置所有步骤为未完成
                for i in range(len(self.workflow_state)):
                    self.workflow_state[i] = (self.workflow_state[i][0], False)
                
                # 重置到第一页并更新显示
                self.current_page_index = 0
                self.update_task_display()
                self.update_page_navigation()
                
                self.display_message("短期记忆已手动清除，所有任务重置为未完成")
            except Exception as e:
                messagebox.showerror("错误", f"清除短期记忆失败: {str(e)}")
        else:
            self.display_message("短期记忆文件不存在")

    def save_workflow_state(self):
        """保存工作流程状态到记忆文件"""
        try:
            # 读取当前memory文件内容
            current_content = ""
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            
            # 分离工作流程定义和执行历史
            lines = current_content.split('\n')
            workflow_lines = []
            history_lines = []
            in_history = False
            
            for line in lines:
                if line == "执行历史:":
                    in_history = True
                    history_lines.append(line)
                    continue
                elif line.startswith("工作流程定义:"):
                    workflow_lines.append(line)
                    continue
                elif line.startswith("任务") and " - " in line and not in_history:
                    workflow_lines.append(line)
                    continue
                elif in_history:
                    history_lines.append(line)
                else:
                    workflow_lines.append(line)
            
            # 更新工作流程状态
            updated_workflow_lines = ["工作流程定义:"]
            for i, (step, completed) in enumerate(self.workflow_state):
                if completed == True:
                    status = "已完成"
                elif completed == "pending_verification":
                    status = "待确定"
                else:
                    status = "待完成"
                updated_workflow_lines.append(f"任务{i+1}: {step} - {status}")
            
            # 合并内容并写回文件
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(updated_workflow_lines))
                f.write('\n\n')
                f.write('\n'.join(history_lines))
        except Exception as e:
            print(f"保存工作流程状态失败: {str(e)}")
def main():
    root = tk.Tk()
    app = VLMTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()