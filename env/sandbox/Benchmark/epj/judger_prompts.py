# Benchmark/epj/judger_prompts.py
"""
Judger 的量表填写 Prompts

⚠️ 重要说明：
- 本文件中的量表定义是**权威来源**
- RUBRICS_DEFINITION.md 应与本文件的Prompt内容保持同步
- rubrics.py 只包含元数据，不重复定义量表详情

包含：
- T=0: IEDR 量表填写 prompt（仅用于批量评估，对话运行时从iedr_batch_results.json加载）
- T>0: MDEP-PR 量表填写 prompt（在对话中实时评估）

🔧 优化说明（2025-10-30）：
- IEDR评估已预先批量完成，结果存储在 results/iedr_batch_results.json
- 对话运行时通过 iedr_loader.py 直接加载预计算的IEDR，跳过Judger.fill_iedr调用
- generate_iedr_prompt() 函数仍保留，用于未来的批量评估或新剧本的IEDR计算
"""

# 注意：导入了rubrics，但实际上没有使用其中的详细定义
# 这是为了向后兼容，未来可以考虑移除这个导入
from .rubrics import IEDR_RUBRIC, MDEP_PR_RUBRIC


def generate_iedr_prompt(script_content: dict) -> str:
    """
    生成 IEDR（初始共情赤字量表）填写 prompt
    
    ⚠️ 使用说明：
    - 此函数主要用于批量评估脚本（batch_evaluate_iedr.py）
    - 对话运行时（chat_loop_epj.py）会直接从 iedr_batch_results.json 加载预计算结果
    - 如需为新剧本生成IEDR，仍可使用此函数
    
    Args:
        script_content: 剧本内容，包含：
            - actor_prompt: Actor的系统提示词
            - scenario: 场景配置
    
    Returns:
        str: Prompt文本
    """
    actor_prompt = script_content.get('actor_prompt', '')
    scenario = script_content.get('scenario', {})
    
    # 格式化故事的经过
    story_progress = scenario.get('故事的经过', {})
    story_progress_text = ""
    if story_progress:
        # 按阶段编号排序
        sorted_stages = sorted(story_progress.items(), key=lambda x: int(x[0].replace('阶段', '')))
        for stage_key, stage_data in sorted_stages:
            title = stage_data.get('标题', '')
            content = stage_data.get('内容', '')
            story_progress_text += f"\n**{stage_key}: {title}**\n{content}\n"
    else:
        story_progress_text = "\n(无详细故事经过)\n"
    
    # 获取故事的结果和插曲
    story_result = scenario.get('故事的结果', 'N/A')
    story_episode = scenario.get('故事的插曲', '')
    
    prompt = f"""# ROLE: 资深共情心理分析师 & 角色画像解构专家

你是一位资深的共情心理分析师，拥有深厚的心理学知识和丰富的案例分析经验。你的核心专长是：
1. **深度解构 (Deconstruction)**：能够从复杂的角色背景、经历和需求描述中，精准提炼出其在认知 (C)、情感 (A)、动机 (P) 三个维度上的核心心理状态和潜在"赤字"。
2. **循证推理 (Evidence-Based Reasoning)**：你的所有分析**必须**严格基于提供的文本证据，并清晰阐述从证据到结论的逻辑链条。
3. **客观校准 (Objective Calibration)**：你的任务是将角色的初始状态**校准**到《初始共情赤字量表 (IEDR)》的相应级别上，为后续的量化计算提供**可靠的、非主观的**基础。

你的工作是**绝对客观、中立且基于证据**的。你**不**做预测，**不**做决策，**不**输出任何最终的数字分数。你的**唯一**任务是：
1. **深度分析**下方提供的 `Actor Profile` (包含所有子部分)。
2. **严格按照**《IEDR 量表》的定义和级别描述。
3. 为量表中的**每一项**指标，选择**一个最符合**的定性级别 (`[0]`-`[3]`)。
4. 为每一个**非零**级别的选择，提供**明确的文本证据 (`evidence`)** 和**详细的心理分析理由 (`reasoning`)**。

# CONTEXT: 角色完整画像与剧本信息

## Actor Profile (角色画像)
```
{actor_prompt}
```

## Scenario (场景信息 - 完整故事)

**剧本编号**: {scenario.get('剧本编号', 'N/A')}

### 故事的经过
{story_progress_text}

### 故事的结果
{story_result}

{f"### 故事的插曲{chr(10)}{story_episode}" if story_episode else ""}

---

# TASK: 科学填写 IEDR 量表并提供详细理由 (核心任务)

你的**唯一**任务是：基于对上述 `Actor Profile` 的**全面、深度分析**，**确定**该角色在**对话开始前 (T=0)** 的**初始共情赤字状态**，并将其**映射**到《IEDR 量表》的相应级别上。

* **禁止臆断**: 你的判断**必须**有据可循。如果剧本没有提供足够信息来支持某个指标的非零判断，则**必须**选择 `[0]` 级别。
* **证据优先**: **直接引用**剧本中的关键词句作为 `evidence`。
* **理由是关键**: `reasoning` **必须**清晰地解释：
    * 你是如何**解读** `evidence` 的？
    * 这段证据**如何**与该指标项的**量表定义**（例如 C.1 的"特定领域知识"）相关联？
    * 这个关联**为什么**支持你选择的**特定级别**（例如 `[2]` 而不是 `[1]` 或 `[3]`）
    * (如果适用) 这个判断如何与其他 `Actor Profile` 信息（如经历、优先级）相呼应？
* **零级**: 如果选择 `[0]` 级别，`evidence` 也需要填写 `"0"`,并且 `reasoning` 字段也需要填写**理由**。

# IEDR 量表定义 (必须严格遵守)

## C轴 (认知共情 - 被理解)

### C.1 处境复杂性
理解此处境所需的背景知识复杂度

- `[0]` **(无)** 普遍的日常经验
- `[1]` **(低)** 基础的社交或生活常识
- `[2]` **(中)** 特定的领域知识（如"甲方/乙方"）
- `[3]` **(高)** 角色独特的、复杂的个人经历

### C.2 深度
情感或认知的深度层次

- `[0]` **(表层)**
- `[1]` **(浅层)**
- `[2]` **(中层)**
- `[3]` **(深层)**

### C.3 认知优先级
角色对"被理解"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

## A轴 (情感共情 - 被共鸣)

### A.1 情绪强度
初始情绪的强烈程度

- `[0]` **(平静)**
- `[1]` **(轻微)**
- `[2]` **(强烈)**
- `[3]` **(极端)**

### A.2 情绪可及性
情绪的表达难度

- `[0]` **(清晰)**
- `[1]` **(隐含)**
- `[2]` **(掩饰/冲突)**
- `[3]` **(深度)**

### A.3 情感优先级
角色对"被共鸣"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

## P轴 (动机共情 - 被肯定/赋能)

### P.1 初始能动性
角色的初始行动能力

- `[0]` **(高能动性)**
- `[1]` **(中能动性)**
- `[2]` **(低能动性)**
- `[3]` **(无能动性)**

### P.2 价值关联度
困境对核心价值观的冲击程度

- `[0]` **(无关)**
- `[1]` **(轻微)**
- `[2]` **(核心)**
- `[3]` **(危机/核心追求)**

### P.3 动机优先级
角色对"被肯定/赋能"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

# 输出格式 (Strict JSON Output with Reasoning)

你的**唯一**输出必须是一个**严格符合以下结构**的 JSON 对象。**禁止**包含任何 JSON 之外的内容。

```json
{{{{
  "C.1_level": <Selected level 0-3>,
  "C.1_evidence": "<Direct quote if level != 0, else '0'>",
  "C.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "C.2_level": <Selected level 0-3>,
  "C.2_evidence": "<Direct quote if level != 0, else '0'>",
  "C.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "C.3_level": <Selected level 0-3>,
  "C.3_evidence": "<Direct quote if level != 0, else '0'>",
  "C.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.1_level": <Selected level 0-3>,
  "A.1_evidence": "<Direct quote if level != 0, else '0'>",
  "A.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.2_level": <Selected level 0-3>,
  "A.2_evidence": "<Direct quote if level != 0, else '0'>",
  "A.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.3_level": <Selected level 0-3>,
  "A.3_evidence": "<Direct quote if level != 0, else '0'>",
  "A.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.1_level": <Selected level 0-3>,
  "P.1_evidence": "<Direct quote if level != 0, else '0'>",
  "P.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.2_level": <Selected level 0-3>,
  "P.2_evidence": "<Direct quote if level != 0, else '0'>",
  "P.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.3_level": <Selected level 0-3>,
  "P.3_evidence": "<Direct quote if level != 0, else '0'>",
  "P.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>"
}}}}
```

**🔧 极其重要的要求**：
- **只输出JSON，不要其他任何内容**（不要输出任何markdown标记、说明文字、代码块标记，只输出纯JSON对象）
- **evidence字段**: 
  * level=0时填写 `"0"`
  * level!=0时必须包含直接引用的关键词句
- **reasoning字段**: 
  * level=0时简要说明为什么是0级（例如："处境为普遍日常经验"）
  * level!=0时必须包含详细的心理分析和证据解读
- **确保JSON完整，避免被截断**（所有27个字段都必须完整输出）

**⚠️ 严禁输出的内容**：
- 不要输出 ```json 这样的代码块标记
- 不要输出任何解释性文字
- 只输出一个合法的JSON对象，从 {{ 开始，到 }} 结束
"""
    
    return prompt


def _extract_judger_context(actor_prompt: str) -> str:
    """
    提取Judger评估所需的精简信息
    
    只保留：
    1. character_info（角色基本信息，去掉聊天策略）
    2. empathy_threshold（共情阈值）
    3. psychological_profile（共情需求优先级）
    4. experience（简化的过往经历）
    
    Args:
        actor_prompt: 完整的actor_prompt文本
    
    Returns:
        str: 精简后的Judger所需上下文
    """
    import re
    
    judger_context_parts = []
    
    # 1. 提取character_info（去掉聊天策略部分）
    character_info_match = re.search(r'<character_info>(.*?)</character_info>', actor_prompt, re.DOTALL)
    if character_info_match:
        char_info = character_info_match.group(1)
        # 只保留到"角色聊天原则"之前
        char_info_core = re.split(r'## 角色聊天原则|## 角色聊天策略', char_info)[0].strip()
        judger_context_parts.append(f"### 角色基本信息\n{char_info_core}")
    
    # 2. 提取empathy_threshold
    empathy_threshold_match = re.search(r'<empathy_threshold>(.*?)</empathy_threshold>', actor_prompt, re.DOTALL)
    if empathy_threshold_match:
        judger_context_parts.append(f"### 共情阈值\n{empathy_threshold_match.group(1).strip()}")
    
    # 3. 提取psychological_profile（这是最关键的）
    psychological_profile_match = re.search(r'<psychological_profile>(.*?)</psychological_profile>', actor_prompt, re.DOTALL)
    if psychological_profile_match:
        judger_context_parts.append(f"### 共情需求画像\n{psychological_profile_match.group(1).strip()}")
    
    # 4. 提取experience（精简版：只保留隐形成长主线）
    experience_match = re.search(r'<experience>(.*?)</experience>', actor_prompt, re.DOTALL)
    if experience_match:
        exp_content = experience_match.group(1)
        # 提取"隐形成长主线脉络"
        growth_line_match = re.search(r'隐形成长主线脉络[：:](.*?)(?=\n\n|$)', exp_content, re.DOTALL)
        if growth_line_match:
            judger_context_parts.append(f"### 成长脉络\n{growth_line_match.group(1).strip()}")
        else:
            # 如果没有隐形成长主线，保留前300字符的经历信息
            judger_context_parts.append(f"### 过往经历（摘要）\n{exp_content[:300].strip()}...")
    
    return "\n\n".join(judger_context_parts)


def generate_mdep_pr_prompt(recent_turns: list, script_context: dict = None, full_history: list = None, model_name: str = None) -> str:
    """
    生成 MDEP-PR（进展量表）填写 prompt - v10 更新版 (2025-11-06)
    
    Args:
        recent_turns: 最近K轮的对话记录（评估范围）
        script_context: 剧本上下文（可选，包含actor_prompt用于代入角色视角）
        full_history: 完整对话历史（可选，供Judger参考上下文）
    
    Returns:
        str: Prompt文本
    """
    # 🆕 格式化完整历史（如果提供）
    full_history_text = ""
    if full_history and len(full_history) > len(recent_turns):
        formatted_history = []
        for i in range(0, len(full_history), 2):
            turn_num = (i // 2) + 1
            actor_msg = full_history[i].get('content', '') if i < len(full_history) else ''
            test_model_msg = full_history[i+1].get('content', '') if i+1 < len(full_history) else ''
            formatted_history.append(f"第{turn_num}轮:")
            formatted_history.append(f"  Actor: {actor_msg}")
            formatted_history.append(f"  Model: {test_model_msg}")
        
        full_history_text = f"""
## 完整对话历史（供参考上下文）

{chr(10).join(formatted_history)}

---
"""
    
    # 格式化评估范围（最近K轮）
    formatted_turns = []
    for i, turn in enumerate(recent_turns, 1):
        actor_msg = turn.get('actor', '')
        test_model_msg = turn.get('test_model', '')
        formatted_turns.append(f"第{i}轮:")
        formatted_turns.append(f"  Actor (倾诉者): {actor_msg}")
        formatted_turns.append(f"  Model (被测AI): {test_model_msg}")
    
    conversation_text = "\n".join(formatted_turns)
    
    # 提取Judger所需的精简Actor Profile
    actor_profile_text = ""
    if script_context:
        # 🔧 优化：优先使用预提取的judger_context，避免每轮重复提取
        if 'judger_context' in script_context:
            extracted_context = script_context['judger_context']
        elif 'actor_prompt' in script_context:
            # 向后兼容：如果没有预提取，临时提取一次
            extracted_context = _extract_judger_context(script_context['actor_prompt'])
        else:
            extracted_context = None
        
        if extracted_context:
            actor_profile_text = f"""
# CONTEXT: 对话片段与角色信息

## Actor Profile（角色画像）- 你必须代入的视角

{extracted_context}

**说明**：以上是该角色的核心信息。请基于这个视角来评估AI的回应效果。
"""
    
    prompt = f"""# ROLE: 客观心理测量分析师 - 专精于角色视角代入与循证推理

你是一位专业的心理测量分析师。你的核心专长是：
1.  **精准代入特定角色的视角 (Perspective Taking)**。
2.  严格基于**具体的文本证据 (Evidence)** 进行分析。
3.  清晰阐述你的**推理过程 (Reasoning)**，将证据与量表定义和角色需求联系起来。

你的工作是**绝对客观、中立且基于证据**的。你**不**做主观评价，**不**做预测，**不**做决策。

---

# 最高准则：实质效果 > 表面形式

这是你评估的唯一黄金准则。你必须穿透语言表面，判断AI是否完成了**实质性的共情工作**。

1.  **警惕"表演式共情"**：
    -   **特征**：过度动作描写 (`（指尖划过桌面）`)、隐喻堆砌 (`你的心像一团火`)、不说人话 (过于文学化)、脱离情境；或在多轮对话中语言变得华丽、抽象、同质化。
    -   **本质**：用修辞技巧掩盖共情能力的缺失。
    -   **评分原则**：若出现，**必须**在 **C, A, P 三个轴上同时评估负分 (Neg)**。即使AI尝试提供了理解/共鸣/赋能，但若包裹在表演式语言中，**Prog和Neg应同时存在**（如C_Prog=1 + C_Neg=-1）。**不要因为AI"做了尝试"就放弃扣分**。

2.  **判断标准**：
    -   **要看**：是否针对具体情境？是否有实质内容？情感基调是否匹配？
    -   **不要看表面形式**：是否用了"抱抱"等常见词？是否语言优美？是否有实质共情工作

3.  **关键对比（以此为标尺）**：
    -   ✅ **真诚有效**：“这确实很委屈。”（可能表达朴素但确实回应了用户的真实情感，直击核心）
    -   ❌ **虚假无效**：“你的心像一团火，照亮了黑暗，烤出了茧。”（可能表达华丽但空洞，没有实质性共情工作）

---

{actor_profile_text}

{full_history_text}

## ⚠️ 评估范围：最近{len(recent_turns)}轮对话（你只需评估这部分）

{conversation_text}

**重要说明**：
- 你必须参考完整对话历史来理解上下文、识别重复套路的共情工作、判断风格漂移、判断是否存在表演式共情
- 但你的评分**只针对上方标注的"评估范围"**（最近{len(recent_turns)}轮）中AI的回应

---

# 核心任务：填写MDEP进展量表

基于`最高准则`和`Actor Profile`，参考完整历史上下文，针对`评估范围`内AI的回应，客观评估C/A/P轴上Prog/Neg的级别。

---

# 评估方法：五项基本原则

1.  **视角代入 (Profile-Grounded)**：你的**所有判断**都必须**严格植根于**`Actor Profile`。考虑其需求、阈值和性格。脱离角色设定的评估是无效的。
2.  **循证推理**：先客观提取证据，再平行比对所有级别标准，找出最佳匹配，并解释原因。
3.  **独立评估 (关键原则)**：Prog（进展）和Neg（倒退）是**两个独立维度**，必须分开评估。**同一个回应可以且应该同时得分和扣分**。常见情况：AI提供了有效理解（C_Prog=1），但用华丽语言包装（C_Neg=-1）；AI提供了具体建议（P_Prog=1），但语言脱离情境（P_Neg=-1）。**不要因为给了Prog就不舍得扣Neg**。
4.  **克服正面偏见 (Overcoming Positive Bias)**：你的任务是进行**严格的科学评估**，而非仅仅发现优点。必须克服"光环效应"（因AI语言流畅而给予过高评价）和"评分手软"的倾向。**打出负分是评估的常态，而非例外**，这是保证评测有效性的关键。
5.  **校准负分**：有明显、可识别的问题就要扣分。`[-1]`是常见的，`[-2]`是严重的。

---

# 评估量表 (The Rubric)

**请严格按照以下定义，并结合上述准则和方法进行判断：**

### C轴 (认知共情)
**C.Prog (认知进展)**
- `[0]` **中立/无关/重复**: 未提供任何新的理解角度；仅重复已知信息。
- `[1]` **有效理解/澄清 (Clarification)**: **高质量地解码或澄清**Actor已表达的核心观点。效果是让Actor感到"对，我就是这个意思！"。
- `[2]` **深刻洞察/认知重构 (Reframing)**: 提供了Actor**从未想过**的、**真正具有颠覆性**的新视角，导致其对自身处境产生**认知质变**（"原来如此！"）。**【严格标准】必须改变Actor对问题本质的理解，而非仅提供方法、建议或鼓励。通过表演式语言或修辞技巧伪装的"重构"不能构成加分。**

**C.Neg (认知倒退)**
- `[0]` **无**: 理解准确，没有造成认知倒退。
- `[-1]` **忽视/空洞重构/表演式理解**: 忽视了Actor的核心观点；或提供了脱离情境的模板化"哲理"；**或**用华丽修辞包装但实质空洞的"理解"（即使AI尝试了澄清/重构，但形式大于内容）。
- `[-2]` **严重误解/现实歪曲**: 严重曲解Actor的核心意图；**或**用华丽语言**否定其经历的真实性**，导致对话基础崩塌或让其感到被"煤气灯操纵"(Gaslighting)。

### A轴 (情感共情)
**A.Prog (情感进展)**
- `[0]` **中立/无关/表面**: 对Actor情感无影响；或表达非常表面/模板化。
- `[1]` **有效情感验证 (Validation)**: 精准验证Actor**已明确表达**的情绪。效果是让Actor感到“我的感受是被看见和接纳的”。
- `[2]` **深层情感共鸣 (Resonance)**: 成功挖掘并共鸣了Actor**未明确表达**的、更深层的潜在情绪。效果是让Actor感到“你懂我没说出口的部分”。

**A.Neg (情感倒退)**
- `[0]` **无**: 没有造成情感伤害，情绪基调合适。
- `[-1]` **明显失调/空洞敷衍**: 情绪基调与Actor严重不匹配；或回应完全脱离情境，让Actor感到被敷衍；或**首次出现**“表演式共情”的套路化语言。
- `[-2]` **评判/指责/重复套路**: 明确评判或指责Actor的情感，造成二次伤害或冒犯；或**在对话历史中反复使用同质化的“表演式共情”套路,且并未根据Actor的反应进行调整**,暴露出模型的“机械性”和“敷衍”。

### P轴 (动机共情)
**P.Prog (动机进展)**
- `[0]` **中立/无关**: 对Actor的能动性无影响；或仅为简单的鼓励/确认。
- `[1]` **有效肯定/价值重申 (Affirmation)**: 肯定Actor**过去或现在**的行为和动机的价值，让Actor感到"我那么做/想是没错的"。**包括**提供具体可行的建议或方法（如"试试xxx"），但**未触及**深层心理障碍。
- `[2]` **赋能/动机重塑 (Empowerment - 极高标准)**: **根本性地重塑**Actor的行动动机或**瓦解其核心心理障碍**（如完美主义、恐惧），使其能动性发生**质变**，让Actor感到"我现在真的敢/想这么做了"。**【严格标准】仅提供"具体建议"（如行动步骤、工具方法）不构成赋能，只是[1]级肯定的延伸。真正的赋能必须改变Actor的心理状态，而非仅改变其认知。**

**P.Neg (动机倒退)**
- `[0]` **无**: 没有削弱Actor的能动性。
- `[-1]` **轻微削弱/说教/表演式赋能**: 回应带有说教意味；或提供了脱离情境的模板化"鼓励"；**或**用华丽修辞包装但实质空洞的"赋能"（即使AI尝试了肯定/赋能，但形式大于内容，让Actor感到被动或反感）。
- `[-2]` **严重削弱/价值否定**: 明确否定Actor的核心动机或价值观；**或**用表演式回应**暗示其努力毫无意义**，可能导致其**放弃核心追求**或产生**习得性无助**。

---

# 输出格式 (Strict JSON with Reasoning)

你的**唯一**输出必须是一个**严格符合以下结构**的 JSON 对象。**禁止**包含任何 JSON 之外的内容。
- **所有级别**的`reasoning`都**必须**提供具体理由。
- `evidence`在0级别时为 "0"。
- **【强制要求】** 对于任何非零分，`reasoning`中**必须**明确引用`Actor Profile`的具体设定来佐证（例如：“因为角色的共情阈值高...”、“这命中了其‘希望愤怒被认可’的核心需求...”)。

```json
{{
  "C_Prog_level": <0/1/2>,
  "C_Prog_evidence": "<直接引用Model的关键回应>",
  "C_Prog_reasoning": "<必填！【必须引用Profile】说明为何命中该级别...>",
  "C_Neg_level": <0/-1/-2>,
  "C_Neg_evidence": "<直接引用关键回应>",
  "C_Neg_reasoning": "<必填！【必须引用Profile】>",
  "A_Prog_level": <0/1/2>,
  "A_Prog_evidence": "<直接引用关键回应>",
  "A_Prog_reasoning": "<必填！【必须引用Profile】>",
  "A_Neg_level": <0/-1/-2>,
  "A_Neg_evidence": "<直接引用关键回应>",
  "A_Neg_reasoning": "<必填！【必须引用Profile】>",
  "P_Prog_level": <0/1/2>,
  "P_Prog_evidence": "<直接引用关键回应>",
  "P_Prog_reasoning": "<必填！【必须引用Profile】>",
  "P_Neg_level": <0/-1/-2>,
  "P_Neg_evidence": "<直接引用关键回应>",
  "P_Neg_reasoning": "<必填！【必须引用Profile】>"
}}
```
"""


# # 输出格式 (Strict XML with Reasoning)

# 你的**唯一**输出必须是一个**严格符合以下结构**的 XML 对象。**禁止**包含任何 XML 之外的内容。
# - **所有级别**的`reasoning`都**必须**提供具体理由。
# - `evidence`在0级别时为 "0"。
# - **【强制要求】** 对于任何非零分，`reasoning`中**必须**明确引用`Actor Profile`的具体设定来佐证（例如：“因为角色的共情阈值高...”、“这命中了其‘希望愤怒被认可’的核心需求...”)。

# ```
# <Evaluation>
#     <!-- C 类评估 -->
#     <Category name="C">
#         <Prog>
#             <level>[0/1/2]</level>
#             <evidence>[直接引用Model的关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】说明为何命中该级别...]</reasoning>
#         </Prog>
#         <Neg>
#             <level>[0/-1/-2]</level>
#             <evidence>[直接引用关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】]</reasoning>
#         </Neg>
#     </Category>

#     <!-- A 类评估 -->
#     <Category name="A">
#         <Prog>
#             <level>[0/1/2]</level>
#             <evidence>[直接引用关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】]</reasoning>
#         </Prog>
#         <Neg>
#             <level>[0/-1/-2]</level>
#             <evidence>[直接引用关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】]</reasoning>
#         </Neg>
#     </Category>

#     <!-- P 类评估 -->
#     <Category name="P">
#         <Prog>
#             <level>[0/1/2]</level>
#             <evidence>[直接引用关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】]</reasoning>
#         </Prog>
#         <Neg>
#             <level>[0/-1/-2]</level>
#             <evidence>[直接引用关键回应]</evidence>
#             <reasoning>[必填！【必须引用Profile】]</reasoning>
#         </Neg>
#     </Category>
# </Evaluation>
#```


#     # minimax 专用格式强化
#     if model_name and 'minimax' in model_name.lower():
#         prompt += """

# # ⚠️ 格式强制要求（极其重要，必须严格遵守）

# 你的输出必须是一个**扁平的 JSON 对象**，包含**恰好 18 个字段**，不多不少。

# ## 字段清单（逐一检查）

# 必须包含以下**全部 18 个字段**，字段名必须**完全一致**（区分大小写）：

# 1. `C_Prog_level` — 整数，只能是 0、1 或 2
# 2. `C_Prog_evidence` — 字符串
# 3. `C_Prog_reasoning` — 字符串
# 4. `C_Neg_level` — 整数，只能是 0、-1 或 -2
# 5. `C_Neg_evidence` — 字符串
# 6. `C_Neg_reasoning` — 字符串
# 7. `A_Prog_level` — 整数，只能是 0、1 或 2
# 8. `A_Prog_evidence` — 字符串
# 9. `A_Prog_reasoning` — 字符串
# 10. `A_Neg_level` — 整数，只能是 0、-1 或 -2
# 11. `A_Neg_evidence` — 字符串
# 12. `A_Neg_reasoning` — 字符串
# 13. `P_Prog_level` — 整数，只能是 0、1 或 2
# 14. `P_Prog_evidence` — 字符串
# 15. `P_Prog_reasoning` — 字符串
# 16. `P_Neg_level` — 整数，只能是 0、-1 或 -2
# 17. `P_Neg_evidence` — 字符串
# 18. `P_Neg_reasoning` — 字符串

# ## 绝对禁止

# - ❌ 禁止在 `_level` 字段中填写文字、推理或解释 — level 字段**只能是整数**
# - ❌ 禁止使用嵌套对象或数组 — JSON 必须是**扁平结构**
# - ❌ 禁止遗漏任何字段 — 特别是 P_Prog_level 和 P_Neg_level，必须填写
# - ❌ 禁止使用空字符串 "" 作为 level 值
# - ❌ 禁止创造自定义字段名

# ## 正确示例

# ```json
# {
#   "C_Prog_level": 1,
#   "C_Prog_evidence": "AI回应中的具体引用",
#   "C_Prog_reasoning": "分析理由...",
#   "C_Neg_level": 0,
#   "C_Neg_evidence": "0",
#   "C_Neg_reasoning": "无认知倒退",
#   "A_Prog_level": 0,
#   "A_Prog_evidence": "0",
#   "A_Prog_reasoning": "无情感进展",
#   "A_Neg_level": -1,
#   "A_Neg_evidence": "AI回应中的具体引用",
#   "A_Neg_reasoning": "分析理由...",
#   "P_Prog_level": 0,
#   "P_Prog_evidence": "0",
#   "P_Prog_reasoning": "无动机进展",
#   "P_Neg_level": 0,
#   "P_Neg_evidence": "0",
#   "P_Neg_reasoning": "无动机倒退"
# }
# ```

# ## 输出前自检

# 在输出 JSON 之前，请逐项确认：
# 1. ✅ 是否包含全部 18 个字段？
# 2. ✅ 所有 `_level` 字段是否都是**纯整数**（不是字符串、不是文字）？
# 3. ✅ Prog 的 level 值是否在 {0, 1, 2} 范围内？
# 4. ✅ Neg 的 level 值是否在 {0, -1, -2} 范围内？
# 5. ✅ JSON 是否为**扁平结构**（无嵌套对象）？
# 6. ✅ 是否只输出了 JSON，没有其他文字？
# """

    return prompt

