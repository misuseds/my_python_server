from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format"
prompt = [{"role": "user", "content": "推荐5个北京的景点。"}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model=model_name,
    quantization="gptq_marlin",
    trust_remote_code=True,
    max_num_batched_tokens=32768,
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
)
sampling_params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=1024, repetition_penalty=1.02)

outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
