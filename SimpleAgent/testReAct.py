import re
import json
from dotenv import load_dotenv
import os
import sys

load_dotenv()
project_path = os.getenv('PYTHONPATH')
if project_path:
    sys.path.insert(0, project_path)

from tool.LLMAgent import Agents
from tool.ToolExecutor import ToolExecutor

'''
    ReAct : 通过先让LLM进行思考（Thought），再决定采取的行动（Action）来解决问题的智能体框架。
    优点 : 
        1、能看到LLM的思考过程，更加透明；可以动态调用工具，适应复杂任务；通过历史记录不断优化决策。 
        2、动态纠错，根据上一步的观察结果，动态调整提示词。 
        3、工具协同，多个工具协同工作，解决单一工具无法完成的复杂任务。
    缺点：
        1、对LLM依赖大，不同的LLM表现差异大；需要精心设计提示词，才能引导出合理的思考和行动；可能出现循环调用工具的情况，需要设置合理的终止条件。
        2、执行效率低，需要多次调用LLM分析。
        3、提示词设计复杂，需要良好的提示词，并且可能不适用于所有LLM。
        4、因为观测结果可能有局部最优。
        5、任务复杂情况下结果不稳定。
'''

# ReAct 默认提示词
REACT_PROMPT_TEMPLATE = """
    请注意，你是一个有能力调用外部工具的智能助手。

    可用工具如下:
    {tools}

    请严格按照以下格式进行回应:

    Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
    Action: 你决定采取的行动，要符合Json格式，必须是以下两种之一:
    - {{'fun': '{{可用工具}}', 'input': '{{工具输入}}'}} : 调用一个可用工具。
    - {{'fun': 'Finish', 'input': '{{最终答案}}'}} : 当你认为已经获得最终答案时。

    现在，请开始解决以下问题:
    Question: {question}
    History: {history} 
    """

class ReActAgent:
    def __init__(self, llm_client: Agents, tool_executor: ToolExecutor, max_steps: int = 5, prompt:str = None):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.prompt = prompt or REACT_PROMPT_TEMPLATE
        self.history = []
        self.process = []

    def run(self, question: str):

        print(f"ReAct智能体开始运行，问题: {question}")
        
        self.process = [] # 每次运行时重置过程记录
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = self.prompt.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("警告:解析Action失败。")
                continue

            # 4. 执行Action
            if tool_name == "Finish":
                # 如果是Finish指令，提取最终答案并结束
                final_answer = tool_input
                print(f"🎉 最终答案: {final_answer}")
                self.history.append(f"FinalAnswer: {final_answer}")
                self.history.append(f"process: {self.process}")
                return self.history
            else:
                # 否则，调用对应工具
                print(f"调用API: {tool_name}({tool_input})")
                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    observation = f"错误:未找到名为 '{tool_name}' 的工具。"
                else:
                    observation = tool_function(tool_input) # 调用真实工具

                print(f"Api调用结果: {observation}")
            
                # 将本轮的结果添加到历史记录中
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")
                self.process.append({"prompt": messages, "thinking": thought, "action": action})

        # 循环结束
        print("已达到最大步数，流程终止。")
        self.history.append(f"process: {self.process}")
        return self.history

    # 解析Thought和Action
    def _parse_output(self, text: str):
        """
            解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    # 解析action
    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        data = json.loads(action_text.replace("'", '"'))
        fun = data.get("fun")
        input_value = data.get("input")
        if fun and input_value:
            return fun, input_value
        return None, None


if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, lambda query: f"模拟搜索结果: {query}")
    
    # 3. 初始化LLM客户端
    llmClient = Agents()

    # 4. 创建ReAct智能体
    agent = ReActAgent(llm_client=llmClient, tool_executor=toolExecutor)

    # 5. 运行智能体，提出一个问题
    question = "华为最新手机型号及主要卖点？"
    history = agent.run(question)

    a = 1