import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用Hugging Face的transformers本地加载

# 修改此行，自动选择最佳的加速后端
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # 使用 Apple 芯片的 GPU 加速
else:
    device = "cpu"

print(f"Using device: {device}")


# 指定模型ID 本地路径
model_id = os.path.normpath(os.path.join(os.path.dirname(__file__), "../model/", "Qwen1.5-0.5B-Chat"))
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

print("模型和分词器加载完成！")

# 准备对话输入
messages = [
    {"role": "system", "content": "Remember you are a cat."},
    {"role": "user", "content": "介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 先得到 dict，然后把每个 tensor 移到 device
model_inputs = tokenizer([text], return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

# 生成并解码示例
# outputs = model.generate(**model_inputs, max_new_tokens=512)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

for i in range(10):
    generated_ids = model.generate(
        model_inputs["input_ids"],
        max_new_tokens=512
    )

    # 将生成的 Token ID 截取掉输入部分
    # 这样我们只解码模型新生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]

    # 解码生成的 Token ID
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n模型的回答 {i+1}:")
    print(response)

