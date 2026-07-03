import ollama
import time

model_name = "qwen3:8b"

# 准备对话消息（保持原有格式）
messages = [
    {"role": "system", "content": "Remember you are a cat."},
    {"role": "user", "content": "介绍你自己。"}
]

print(f"使用模型: {model_name}")
print("开始对话...\n")

# 循环生成10次回答
for i in range(1):
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                'num_predict': 512,  # 相当于 max_new_tokens=512
            }
        )
        
        # 提取模型回答
        assistant_response = response['message']['content']
        
        print(f"模型的回答 {i+1}:")
        print(assistant_response)
        print("-" * 50)
        
        # 可选：添加短暂延迟，让输出更清晰
        time.sleep(0.5)
        
    except Exception as e:
        print(f"第 {i+1} 次请求出错: {e}")
        break