import os
import re

from BaseAgent.OpenAiClient import OpenAICompatibleClient
from BaseAgent.get_attraction import get_attraction
from BaseAgent.get_weather import get_weather

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
# API_KEY = "API_KEY_PLACEHOLDER"
# BASE_URL = "BASE_URL_PLACEHOLDER"
# MODEL_ID = "MODEL_ID_PLACEHOLDER"
# TAVILY_API_KEY="YOUR_Tavily_KEY"
# os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_API_KEY"

model = ""

llm = OpenAICompatibleClient(model=model)

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

AGENT_SYSTEM_PROMPT = '如果未解决问题：格式要求: Thought: ... Action: ...，并且如果问题解决Action: Finish[答案]' \

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    
    # 3.3. 解析回答格式
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        prompt_history.append(observation_str)
        continue

    # 3.4. 模型输出
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            # print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.5. 匹配是否完成任务
    action_str = action_match.group(1).strip()
    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    # 3.6. 解析工具调用 todo
    tool_name = re.search(r"(\w+)\(", action_str)
    args_str = re.search(r"\((.*)\)", action_str)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name.group(1) in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.7. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
