# Benchmark/agents/test_model.py (修复版)

from Benchmark.llms.api import get_llm_response
from Benchmark.prompts.test_model_prompts import generate_test_model_prompts

class TestModel:
    """
    代表被测评的模型的智能体。
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_reply(self, history: list) -> str:
        """
        根据对话历史生成回复。
        """
        # 获取system和user prompts
        system_prompt, user_prompt = generate_test_model_prompts(history)
        
        # 🔧 修复: 只构建基础messages，不重复添加历史
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 注：历史记录已经在user_prompt中包含，不需要重复添加

        try:
            print(f"--- [TestModel] 正在生成共情回复（模型: {self.model_name}）... ---")
            
            reply = get_llm_response(messages=messages, model_name=self.model_name)
            
            # 检查回复长度并提示
            word_count = len(reply)
            if word_count > 500:
                print(f"⚠️ [TestModel] 警告：回复过长（{word_count}字），可能违反了简洁性原则")
            
            return reply
            
        except Exception as e:
            print(f"!!! [TestModel] 调用LLM API时发生严重错误: {e} !!!")
            return "抱歉，我在处理你的请求时遇到了一个问题。"