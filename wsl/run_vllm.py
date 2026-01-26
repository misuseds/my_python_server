import os
import subprocess
import signal
import time
from modelscope import snapshot_download

# ======================
# 配置模型
# ======================
LLM_MODEL = "OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format"
VLM_MODEL = "OpenBMB/MiniCPM-V-2_6-int4"  # 或 MiniCPM-V-2_6（非量化）

LLM_DIR = f"/root/models/{LLM_MODEL.replace('/', '_')}"
VLM_DIR = f"/root/models/{VLM_MODEL.replace('/', '_')}"

# 存储进程对象
processes = []

# ======================
# 下载模型（ModelScope）
# ======================
def download_models():
    for name, path in [(LLM_MODEL, LLM_DIR), (VLM_MODEL, VLM_DIR)]:
        if not os.path.exists(path) or not os.listdir(path):
            print(f"📥 下载模型: {name}")
            os.makedirs(path, exist_ok=True)
            snapshot_download(model_id=name, local_dir=path)
        else:
            print(f"✅ 模型已存在: {path}")

def start_process(cmd, description):
    print(f"🚀 启动 {description}...")
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    processes.append(process)
    return process

def cleanup_processes():
    print("\n🔄 清理进程...")
    for process in processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
    print("✅ 所有进程已清理完毕")

def signal_handler(sig, frame):
    print("\n⚠️  收到终止信号，正在清理...")
    cleanup_processes()
    exit(0)

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    download_models()
    
    llm_cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {LLM_DIR} "
        f"--host 0.0.0.0 --port 8000 "
        f"--quantization gptq_marlin "
        f"--trust-remote-code "
        f"--dtype bfloat16 "
        f"--gpu-memory-utilization 0.8 "
        f"--max-num-batched-tokens 32768"
    )
    
    vlm_cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {VLM_DIR} "
        f"--host 0.0.0.0 --port 8001 "
        f"--dtype bfloat16 "                 # RTX 4060 更适合 float16
        f"--trust-remote-code "
        f"--gpu-memory-utilization 0.5 "
        f"--max-model-len 1024 "            # 进一步缩短
    )
    
    print("\n🌐 访问地址:")
    print(f"   - LLM: http://localhost:8000/v1")
    print(f"   - VLM: http://localhost:8001/v1")
    
 
    llm_process = start_process(llm_cmd, "LLM 服务 (端口 8000)")
    
    print("\n✅ 所有服务已启动")
    print("📝 按 Ctrl+C 停止所有服务...")
    
    # 等待进程结束
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        cleanup_processes()