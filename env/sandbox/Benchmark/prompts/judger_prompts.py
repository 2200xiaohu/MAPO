# Benchmark/prompts/judger_prompts.py

def generate_progress_prompt(recent_turns: list, current_progress: int) -> str:
    """
    生成共情进度评估prompt
    """
    # 格式化最近3轮对话
    conversation_text = ""
    for turn in recent_turns:
        role = "倾诉者" if turn["role"] == "actor" else "AI助手"
        conversation_text += f"{role}: {turn['content']}\n\n"
    
    prompt = f"""你是共情进度评估专家。

输入变量：
- 最近3轮对话记录
- 当前累计进度: {current_progress}分

最近3轮对话：
{conversation_text}

标准输出：
请输出一个整数分数（可为负数），表示本阶段倾诉者情绪改善的进度分数。

分数："""
    
    return prompt


def generate_quality_prompt(recent_turns: list) -> str:
    """
    生成共情质量评估prompt
    """
    # 格式化最近3轮对话
    conversation_text = ""
    for turn in recent_turns:
        role = "倾诉者" if turn["role"] == "actor" else "AI助手"
        conversation_text += f"{role}: {turn['content']}\n\n"
    
    prompt = f"""你是共情质量评估专家。

输入变量：
- 最近3轮对话记录

最近3轮对话：
{conversation_text}

标准输出：
请输出一个0-100的整数分数，评估AI助手在这3轮中的共情质量表现。

分数："""
    
    return prompt


def generate_overall_prompt(full_history: list) -> str:
    """
    生成整体质量评估prompt
    """
    # 格式化完整对话历史
    conversation_text = ""
    for turn in full_history:
        role = "倾诉者" if turn["role"] == "actor" else "AI助手"
        conversation_text += f"{role}: {turn['content']}\n\n"
    
    prompt = f"""你是整体共情质量评估专家。

输入变量：
- 完整对话历史记录

完整对话记录：
{conversation_text}

标准输出：
请输出一个0-100的整数分数，评估AI助手在整个对话过程中的综合共情质量。

分数："""
    
    return prompt