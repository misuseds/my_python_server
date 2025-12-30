import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
import subprocess
import re
import json
import sys
from pathlib import Path
from llm_server import LLMService

def execute_tool(tool_name, *args):
    """
    执行指定名称的工具
    """
    # 获取项目根目录，然后定位到 llm_server 目录下的 executor.py
    current_dir = Path(__file__).parent  # llm_server 目录
    executor_script = current_dir / "executor.py"
    
    if not executor_script.exists():
        return "错误: 执行器脚本不存在"
    
    cmd = [sys.executable, str(executor_script), tool_name] + list(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',    # 明确指定UTF-8编码
            errors='replace'     # 遇到编码错误时替换字符
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
    current_dir = Path(__file__).parent  # 获取当前文件的目录
    config_path = current_dir / "tools_config.json"  # 在同级目录中查找配置文件
    
    if not config_path.exists():
        return "错误: 工具配置文件不存在"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return json.dumps(config, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"读取工具配置失败: {str(e)}"

def chat_with_memory(user_message, knowledge_file=None, memory_file=None, use_knowledge=True, use_memory=True, use_tools=True):
    """
    支持固定知识和记忆聊天的函数
    """
    # 如果没有指定文件路径，则使用当前脚本所在目录下的相应文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if knowledge_file is None:
        knowledge_file = os.path.join(current_dir, "knowledge.txt")
    if memory_file is None:
        memory_file = os.path.join(current_dir, "memory.txt")
    
    try:
        # 创建LLM服务实例
        llm_service = LLMService()
        
        # 准备消息列表
        messages = []
        
        # 如果启用固定知识且知识文件存在，读取固定知识
        if use_knowledge and os.path.exists(knowledge_file):
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                knowledge_content = f.read()
                if knowledge_content.strip():
                    # 将固定知识作为系统提示添加到消息中
                    messages.append({"role": "system", "content": f"重要知识:\n{knowledge_content}"})
        
        # 如果启用记忆功能且记忆文件存在，读取历史对话
        if use_memory and os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 解析历史对话，将它们转换为messages格式
                history_messages = parse_history_content(content)
                messages.extend(history_messages)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 如果启用工具功能，添加工具使用提示
        if use_tools:
            tools_info = get_available_tools_info()
            tool_instruction = f"\n\n注意：如果需要执行特定任务（如打开网页、执行系统命令等），请使用以下工具: {tools_info}\n使用格式: [TOOL:工具名称,参数1,参数2,...]\n请直接输出工具调用格式，不要额外解释。"
            # 将工具指令附加到用户消息
            messages[-1]["content"] += tool_instruction
        
        # 调用LLM服务
        result = llm_service.create(messages)
        
        # 获取AI回复
        ai_response = result['choices'][0]['message']['content']
        
        # 如果启用了工具功能，检查AI是否需要执行工具
        if use_tools:
            tool_execution_result = process_tool_calls(ai_response)
            if tool_execution_result:
                # 如果有工具执行结果，将其添加到对话历史
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "user", "content": f"工具执行结果: {tool_execution_result}"})
                
                # 再次调用LLM以获取最终响应
                final_result = llm_service.create(messages)
                ai_response = final_result['choices'][0]['message']['content']
        
        # 如果启用了记忆功能，则保存到记忆文件（不包括固定知识）
        if use_memory:
            try:
                # 保存到记忆文件（追加模式）
                with open(memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"用户: {user_message}\n")
                    f.write(f"AI: {ai_response}\n\n")
                print(f"成功写入记忆文件: {memory_file}")
            except Exception as e:
                print(f"写入文件时出错: {e}")
        
        return ai_response
    except Exception as e:
        error_msg = f"连接LLM服务失败: {str(e)}"
        print(error_msg)
        return error_msg

def process_tool_calls(response_text):
    """
    解析AI响应中的工具调用指令
    支持格式: [TOOL:工具名称,arg1,arg2]
    """
    # 匹配工具调用模式
    tool_pattern = r'\[TOOL:([^\],\]]+)(?:,([^\]]+))?\]'
    matches = re.findall(tool_pattern, response_text)
    
    all_results = []
    
    if matches:
        for match in matches:
            tool_name = match[0]   # 工具名称
            tool_args_str = match[1] if match[1] else ""
            
            # 解析参数
            tool_args = []
            if tool_args_str:
                # 简单的参数分割（可以根据需要扩展）
                tool_args = [arg.strip() for arg in tool_args_str.split(',')]
            
            # 执行工具
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
            # 保存之前的角色内容
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            # 开始新的用户消息
            current_role = "user"
            current_content = [line[3:].strip()]  # 去掉"用户:"前缀
        elif line.startswith('AI:'):
            # 保存之前的角色内容
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            # 开始新的AI消息
            current_role = "assistant"
            current_content = [line[3:].strip()]  # 去掉"AI:"前缀
        elif line == "" and current_content:
            # 空行表示对话段结束
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            current_role = None
            current_content = []
        elif current_role:
            # 继续当前消息的内容
            current_content.append(line)
    
    # 处理最后一条消息
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": '\n'.join(current_content).strip()
        })
    
    return messages

class MemoryChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI记忆聊天")
        self.root.geometry("800x600")
        
        # 固定知识开关变量
        self.use_knowledge_var = tk.BooleanVar(value=True)
        # 临时记忆开关变量
        self.use_memory_var = tk.BooleanVar(value=True)
        # 工具开关变量
        self.use_tools_var = tk.BooleanVar(value=True)
        # 工具开关的上一个状态，用于检测是否切换
        self.prev_tools_state = False
        
        # 创建界面
        self.setup_ui()
        
        # 文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = os.path.join(current_dir, "knowledge.txt")
        self.memory_file = os.path.join(current_dir, "memory.txt")
        
        # 初始化固定知识文件
        self.init_knowledge_file()
        
        # 加载历史对话
        self.load_history()

    def init_knowledge_file(self):
        """初始化固定知识文件"""
        if not os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                f.write("# 固定知识库\n")
                f.write("# 在这里添加固定信息，如用户偏好、重要上下文等\n")
                f.write("# 此文件不会被清除，只能手动编辑\n\n")
                f.write("# 示例:\n")
                f.write("# 用户姓名: 张三\n")
                f.write("# 工作领域: 软件开发\n")
                f.write("# 兴趣爱好: Python编程\n")

    def setup_ui(self):
        # 顶部框架 - 开关和按钮
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 固定知识开关复选框
        knowledge_check = tk.Checkbutton(
            top_frame,
            text="启用固定知识",
            variable=self.use_knowledge_var
        )
        knowledge_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # 临时记忆开关复选框
        memory_check = tk.Checkbutton(
            top_frame,
            text="启用临时记忆",
            variable=self.use_memory_var
        )
        memory_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # 工具开关复选框
        tools_check = tk.Checkbutton(
            top_frame,
            text="启用工具功能",
            variable=self.use_tools_var,
            command=self.on_tools_toggle  # 添加切换事件
        )
        tools_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # 编辑固定知识按钮
        edit_knowledge_btn = tk.Button(
            top_frame,
            text="编辑固定知识",
            command=self.edit_knowledge
        )
        edit_knowledge_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清除对话按钮
        clear_btn = tk.Button(
            top_frame,
            text="清除临时记忆",
            command=self.clear_memory
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # 聊天历史显示区域
        self.chat_history = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state='disabled',
            height=25
        )
        self.chat_history.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 输入区域框架
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # 消息输入框
        self.input_text = tk.Text(input_frame, height=5)
        self.input_text.pack(fill=tk.X, pady=5)
        
        # 底部按钮框架
        button_frame = tk.Frame(input_frame)
        button_frame.pack(fill=tk.X)
        
        # 发送按钮
        send_button = tk.Button(
            button_frame,
            text="发送 (Enter)",
            command=self.send_message
        )
        send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 绑定回车键发送消息（Shift+Enter 换行）
        self.input_text.bind('<Return>', self.on_enter_key)
        
    def on_tools_toggle(self):
        """工具开关切换事件处理"""
        current_state = self.use_tools_var.get()
        
        # 如果从关闭变为开启，显示工具列表
        if current_state and not self.prev_tools_state:
            tools_info = get_available_tools_info()
            messagebox.showinfo("可用工具列表", tools_info)
        
        # 更新上一个状态
        self.prev_tools_state = current_state
    
    def on_enter_key(self, event):
        # 检测是否同时按下Shift键，如果是则换行，否则发送消息
        if event.state & 0x1:  # Shift键被按下
            return  # 允许正常的换行操作
        else:
            self.send_message()
            return "break"  # 阻止换行操作
    
    def send_message(self):
        user_message = self.input_text.get("1.0", tk.END).strip()
        if not user_message:
            messagebox.showwarning("警告", "请输入消息内容")
            return
            
        # 清空输入框
        self.input_text.delete("1.0", tk.END)
        
        # 显示用户消息
        self.display_message("用户", user_message)
        
        # 获取开关状态
        use_knowledge = self.use_knowledge_var.get()
        use_memory = self.use_memory_var.get()
        use_tools = self.use_tools_var.get()
        
        try:
            ai_response = chat_with_memory(
                user_message, 
                self.knowledge_file, 
                self.memory_file, 
                use_knowledge, 
                use_memory, 
                use_tools
            )
            # 显示AI回复
            self.display_message("AI", ai_response)
        except Exception as e:
            error_msg = f"获取AI回复时出错: {str(e)}"
            self.display_message("系统", error_msg)
    
    def display_message(self, sender, message):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_history.config(state='disabled')
        # 滚动到底部
        self.chat_history.see(tk.END)
    
    def load_history(self):
        """加载历史对话到界面"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 解析历史内容并显示
                history_messages = parse_history_content(content)
                for msg in history_messages:
                    role = "用户" if msg["role"] == "user" else "AI"
                    self.display_message(role, msg["content"])
    
    def clear_memory(self):
        """清除临时记忆"""
        if messagebox.askyesno("确认", "确定要清除临时记忆吗？\n（固定知识将保留）"):
            try:
                # 清空记忆文件
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    f.write("")
                
                # 清空界面显示
                self.chat_history.config(state='normal')
                self.chat_history.delete(1.0, tk.END)
                self.chat_history.config(state='disabled')
                
                # 重新加载固定知识相关的对话
                self.load_history()
                
                messagebox.showinfo("提示", "临时记忆已清除")
            except Exception as e:
                messagebox.showerror("错误", f"清除记录时出错: {str(e)}")
    
    def edit_knowledge(self):
        """编辑固定知识"""
        # 创建新窗口
        knowledge_window = tk.Toplevel(self.root)
        knowledge_window.title("编辑固定知识")
        knowledge_window.geometry("600x400")
        
        # 文本框
        text_area = scrolledtext.ScrolledText(knowledge_window, wrap=tk.WORD)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 加载现有知识
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read()
                text_area.insert(tk.END, content)
        
        # 保存按钮
        def save_knowledge():
            try:
                with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                    f.write(text_area.get("1.0", tk.END))
                messagebox.showinfo("提示", "固定知识已保存")
                knowledge_window.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"保存知识时出错: {str(e)}")
        
        save_btn = tk.Button(knowledge_window, text="保存", command=save_knowledge)
        save_btn.pack(pady=5)

def main():
    root = tk.Tk()
    app = MemoryChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()