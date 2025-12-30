import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
from llm_server import LLMService

def chat_with_memory(user_message, memory_file=None, use_memory=True):
    """
    简单的记忆聊天函数，将对话保存到文件
    :param user_message: 用户消息
    :param memory_file: 记忆文件路径
    :param use_memory: 是否使用记忆功能
    :return: AI回复内容
    """
    # 如果没有指定记忆文件路径，则使用当前脚本所在目录下的memory.txt
    if memory_file is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        memory_file = os.path.join(current_dir, "memory.txt")
    
    try:
        # 创建LLM服务实例
        llm_service = LLMService()
        
        # 准备消息列表
        messages = []
        
        # 如果启用记忆功能且记忆文件存在，读取历史对话
        if use_memory and os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 这里需要解析历史对话，将它们转换为messages格式
                # 简单示例：将历史对话按行分割并转换为消息列表
                history_messages = parse_history_content(content)
                messages.extend(history_messages)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 调用LLM服务
        result = llm_service.create(messages)
        
        # 获取AI回复
        ai_response = result['choices'][0]['message']['content']
        
        # 如果启用了记忆功能，则保存到记忆文件
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
        
        # 记忆开关变量
        self.use_memory_var = tk.BooleanVar(value=True)
        
        # 创建界面
        self.setup_ui()
        
        # 记忆文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.memory_file = os.path.join(current_dir, "memory.txt")
        
        # 加载历史对话
        self.load_history()

    def setup_ui(self):
        # 顶部框架 - 记忆开关和清除按钮
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 记忆开关复选框
        memory_check = tk.Checkbutton(
            top_frame,
            text="启用记忆功能",
            variable=self.use_memory_var
        )
        memory_check.pack(side=tk.LEFT)
        
        # 清除对话按钮
        clear_btn = tk.Button(
            top_frame,
            text="清除对话记录",
            command=self.clear_history
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
        
        # 获取AI回复
        use_memory = self.use_memory_var.get()
        
        try:
            ai_response = chat_with_memory(user_message, self.memory_file, use_memory)
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
    
    def clear_history(self):
        """清除对话记录"""
        if messagebox.askyesno("确认", "确定要清除所有对话记录吗？"):
            try:
                # 清空记忆文件
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    f.write("")
                
                # 清空界面显示
                self.chat_history.config(state='normal')
                self.chat_history.delete(1.0, tk.END)
                self.chat_history.config(state='disabled')
                
                messagebox.showinfo("提示", "对话记录已清除")
            except Exception as e:
                messagebox.showerror("错误", f"清除记录时出错: {str(e)}")

def main():
    root = tk.Tk()
    app = MemoryChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()