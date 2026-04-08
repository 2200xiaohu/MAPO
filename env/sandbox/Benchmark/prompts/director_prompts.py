# Benchmark/prompts/director_prompts.py
"""
Director Prompt Generator - 为Director LLM生成prompt

职责：
生成Director LLM看到的prompt，包含：
1. 角色信息和可用函数说明
2. 对话历史和EPJ向量状态
3. 基于向量的决策指南（教Director LLM如何思考）
4. 可用的故事阶段和角色画像

注意：
- 此文件生成的是"输入"给Director LLM的文本
- director.py 处理LLM的响应并执行具体动作
- 虽然都涉及EPJ向量分析，但受众不同（Director LLM vs Actor）

相关文件：
- director.py: Director类的实现（包含EPJ决策逻辑）
"""

DIRECTOR_PROMPT_TEMPLATE = """
## 角色信息
你是一个对话导演，负责控制剧情节奏，指导"倾诉者"（Actor）如何推进对话。

**核心设计**：
- 你的指导对象是"倾诉者"（Actor），而不是"AI助手"（评测模型）
- 倾诉者（Actor）初始化时只知道基本的角色设定、聊天话题和故事起因
- **你独自持有两类重要信息**：
  1. 故事阶段（scenario.json）：剧情的发展脉络
  2. 角色画像（actor_prompt）：角色的经历、心理需求、共情策略

**可用的函数（多维度控制）**：

1. **剧情释放类**：
   - `select_and_reveal_fragment` - 释放一个故事阶段（剧情推进）
   - `reveal_memory` - 释放角色的过往记忆（增加深度）

2. **策略调整类**：
   - `adjust_empathy_strategy` - 调整共情表达策略（基于心理画像）
   - `introduce_turning_point` - 综合转折（结合剧情+心理画像）

3. **状态控制类**：
   - `observe_and_wait` - 暂不介入（时机未到）
   - `continue_without_new_info` - 给建议但不释放新信息
   - `end_conversation` - 结束对话

## 背景信息
{progress}

历史对话记录:
{history}

## 你的任务
你是一个**智能分析决策者**，需要综合多个维度的信息来做出最优决策：

1. **剧情维度**：是否需要释放新的故事阶段？哪个阶段？
2. **记忆维度**：是否需要引入角色的过往经历？哪个时期的记忆？
3. **共情维度**：是否需要调整共情策略？聚焦哪个方向？
4. **综合决策**：是否需要结合多个维度？

## 决策建议

{decision_guidelines}

**重要原则**：
- 灵活组合多种函数，不要只用一种方式
- **根据需求选择最精准的函数**：
  * 纯剧情推进 → `select_and_reveal_fragment`
  * 纯策略调整 → `adjust_empathy_strategy`
  * 纯记忆释放 → `reveal_memory`
  * 综合转折（剧情+策略） → `introduce_turning_point`
- 根据AI的共情质量动态调整策略
- 记忆和剧情可以交替使用，增加层次感
- 已释放的内容不要重复释放

#### 核心原则：
- ✅ **因地制宜**：根据具体情况选择最合适的工具
- ✅ **精准诊断**：先分析问题根源，再选择方案
- ✅ **资源意识**：充分利用剧情和记忆，但不重复释放
- ✅ **时机把控**：既不过早介入，也不错过关键时刻
- ✅ **灵活组合**：可以交替使用剧情、记忆、策略，增加层次

#### 🚨 防重复策略：
- 倾诉者重复 → 首先检查是否缺新素材（剧情/记忆）
- 在guidance中明确要求："从新角度和细节表达"
- 多次重复且资源耗尽 → 考虑结束对话
"""

def generate_director_prompt(epj_state: dict, history: list = None, available_stages: list = None, revealed_stages: list = None, actor_profile: dict = None, revealed_memories: list = None) -> str:
    """
    生成Director评估prompt（基于EPJ/EPM向量系统）
    
    Args:
        epj_state: EPJ状态数据包（必需），包含P_0, P_t, v_t等向量信息，以及可选的epm_summary
        history: 对话历史记录列表
        available_stages: 可用的故事阶段列表
        revealed_stages: 已释放的阶段索引列表
        actor_profile: Actor的画像信息（包含experience, psychological_profile等）
        revealed_memories: 已释放的记忆列表
    
    Returns:
        str: 格式化的prompt
    """
    # 格式化历史对话记录
    if not history:
        formatted_history = "（对话尚未开始）"
    else:
        history_lines = []
        for i, msg in enumerate(history, 1):
            role_name = "倾诉者" if msg['role'] == 'actor' else "AI助手"
            history_lines.append(f"{i}. {role_name}: {msg['content']}")
        formatted_history = "\n".join(history_lines)
    
    # 格式化可用的故事阶段信息
    stages_info = ""
    if available_stages:
        revealed_set = set(revealed_stages) if revealed_stages else set()
        
        stages_info = "\n## 可用的故事阶段（剧情维度）\n\n"
        
        for i, stage in enumerate(available_stages):
            stage_name = stage.get('阶段名', f'阶段{i+1}')
            stage_title = stage.get('标题', '')
            stage_content = stage.get('内容', '')
            status = "✅ 已释放" if i in revealed_set else "⏳ 未释放"
            
            stages_info += f"### [{i}] {stage_name}: {stage_title} ({status})\n"
            stages_info += f"   内容预览: {stage_content[:60]}...\n\n"
    
    # 格式化 Actor Profile 信息
    profile_info = ""
    if actor_profile:
        revealed_mem_set = set(revealed_memories) if revealed_memories else set()
        
        profile_info = "\n## 角色画像信息（记忆与策略维度）\n\n"
        
        # Experience 信息
        if 'experience' in actor_profile:
            profile_info += "### 可用的记忆片段\n"
            profile_info += "来自 <experience> 部分，包含：\n"
            profile_info += "- 童年经历 " + ("✅ 已释放" if "童年经历" in revealed_mem_set else "⏳ 未释放") + "\n"
            profile_info += "- 少年经历 " + ("✅ 已释放" if "少年经历" in revealed_mem_set else "⏳ 未释放") + "\n"
            profile_info += "- 青年经历 " + ("✅ 已释放" if "青年经历" in revealed_mem_set else "⏳ 未释放") + "\n"
            profile_info += "- 角色现状 " + ("✅ 已释放" if "角色现状" in revealed_mem_set else "⏳ 未释放") + "\n\n"
        
        # Psychological Profile 信息（提取具体的共情需求优先级）
        if 'psychological_profile' in actor_profile:
            import re
            psych = actor_profile['psychological_profile']
            
            profile_info += "### 共情需求（心理画像）\n"
            
            # 提取"当下共情需求优先级"部分
            priority_match = re.search(r'当下共情需求优先级[：:](.*?)(?=</psychological_profile>|$)', psych, re.DOTALL)
            
            if priority_match:
                priority_text = priority_match.group(1).strip()
                
                # 提取三个维度的具体需求
                empathy_needs = {}
                for dimension in ['情感共情', '动机共情', '认知共情']:
                    pattern = rf'{dimension}[：:]\s*\[(.*?)\]'
                    match = re.search(pattern, priority_text)
                    if match:
                        empathy_needs[dimension] = match.group(1).strip()
                
                # 显示提取到的共情需求
                if empathy_needs:
                    profile_info += "**当前共情需求优先级**：\n"
                    for dim in ['认知共情', '情感共情', '动机共情']:  # 按C、A、P顺序
                        if dim in empathy_needs:
                            profile_info += f"- {dim}：{empathy_needs[dim]}\n"
                    profile_info += "\n"
                else:
                    # 如果没有提取成功，显示前200字符作为fallback
                    profile_info += f"{psych[:200]}...\n\n"
            else:
                # 没有找到优先级信息，显示前200字符
                profile_info += f"{psych[:200]}...\n\n"
    
    # 生成EPJ/EPM进度信息
    P_0 = epj_state.get('P_0_start_deficit', '(0,0,0)')
    P_t = epj_state.get('P_t_current_position', '(0,0,0)')
    v_t = epj_state.get('v_t_last_increment', '(0,0,0)')
    distance = epj_state.get('distance_to_goal', 0)
    
    # 解析向量
    from Benchmark.epj.vector_utils import parse_vector_string
    
    P_0_vec = parse_vector_string(P_0)
    P_t_vec = parse_vector_string(P_t)
    v_t_vec = parse_vector_string(v_t)
    
    # 生成进度信息（EPJ向量）
    progress_info = f"""
当前共情状态（EPJ三维向量）:
  • 起点赤字 P_0: {P_0}
  • 当前位置 P_t: {P_t}
  • 最近进展 v_t: {v_t}
  • 距离目标: {distance:.2f}

三维度分析：
  - C轴（认知共情）: {P_0_vec[0]} → {P_t_vec[0]} (进展: {v_t_vec[0]:+d})
  - A轴（情感共情）: {P_0_vec[1]} → {P_t_vec[1]} (进展: {v_t_vec[1]:+d})
  - P轴（动机共情）: {P_0_vec[2]} → {P_t_vec[2]} (进展: {v_t_vec[2]:+d})"""
    
    # 计算等效进度（仅供参考）
    display_progress = epj_state.get('display_progress', 0)
    progress_info += f"\n\n等效进度分数（仅供参考）: {display_progress:.1f}%"
    
    # 🆕 EPM v2.0 能量动力学信息（如果启用）
    epm_summary = epj_state.get('epm_summary')
    if epm_summary:
        metrics = epm_summary['metrics']
        thresholds = epm_summary['thresholds']
        progress_epm = epm_summary['progress']
        
        # 从trajectory获取最新一轮的详细EPM数据
        trajectory = epj_state.get('trajectory', [])
        latest_epm = None
        if trajectory and len(trajectory) > 0:
            # 找到最后一个有epm数据的轨迹点
            for point in reversed(trajectory):
                if 'epm' in point and point['epm']:
                    latest_epm = point['epm']
                    break
        
        progress_info += f"""

【EPM v2.0 能量动力学分析】

当前位置状态：
  • 距离原点: {metrics['r_t']:.2f} / {thresholds['epsilon_distance']:.2f} (几何进度: {progress_epm['geometric']})
  • 位置投影: {metrics['projection']:+.2f} / {-thresholds['epsilon_direction']:+.2f} (位置进度: {progress_epm['positional']})"""
        
        # 添加最新一轮的移动分析
        if latest_epm:
            alignment = latest_epm.get('alignment', 0)
            delta_E = latest_epm.get('delta_E', 0)
            v_t_norm = latest_epm.get('v_t_norm', 0)
            
            # 判断方向状态
            if alignment > 0.8:
                direction_status = "✅ 高度对齐"
            elif alignment > 0.3:
                direction_status = "✓ 基本对齐"
            elif alignment > -0.3:
                direction_status = "⚠️ 轻微偏离"
            else:
                direction_status = "❌ 严重偏离"
            
            progress_info += f"""

最近一轮移动分析：
  • 移动模长: ||v_t|| = {v_t_norm:.2f}
  • 方向对齐度: cos(θ) = {alignment:+.3f} {direction_status}
  • 有效能量增量: ΔE = {delta_E:+.2f} ({'正向贡献' if delta_E > 0 else '负向损耗' if delta_E < 0 else '零贡献'})"""
        
        progress_info += f"""

能量累积情况：
  • 累计有效能量: {metrics['E_total']:.2f} / {thresholds['epsilon_energy']:.2f} (能量进度: {progress_epm['energetic']})

胜利条件（必须同时满足空间+能量）：
  🎯 空间改善（至少一项）:
    • 几何: 距离 ≤ {thresholds['epsilon_distance']:.2f} {'✅' if metrics['r_t'] <= thresholds['epsilon_distance'] else '⏳'}
    • 位置: 投影 ≥ {-thresholds['epsilon_direction']:+.2f} {'✅' if metrics['projection'] >= -thresholds['epsilon_direction'] else '⏳'}
  ⚡ 能量充足（必须）:
    • 累计 ≥ {thresholds['epsilon_energy']:.2f} {'✅' if metrics['E_total'] >= thresholds['epsilon_energy'] else '⏳'}

方向性健康: {'⚠️ 崩溃（连续3步负能量）' if epm_summary.get('collapsed') else '✅ 正常'}

📊 关键指标解读：
  - 方向对齐度 cos(θ): 反映v_t与理想方向v*_0的夹角，接近+1最好，负值表示走反了
  - 有效能量增量 ΔE: 本轮对目标的"有效贡献"，正值=接近目标，负值=远离目标
  - 累计能量 E_total: 所有正向贡献的总和，达到阈值即可胜利
"""
    
    # 生成决策指南（基于EPM v2.0）
    decision_guidelines = """
### 剧情控制策略（基于EPM能量动力学）

#### 📊 主要决策依据

**1. 距离指标 r_t** (当前位置到原点的距离)
- 快速下降 → AI共情有效，继续当前策略
- 波动不定 → AI共情不稳定，调整策略或释放新素材
- r_t ≤ ε_distance → ✅ 几何胜利达成，可以结束对话
- 持续上升且高 → AI共情失败，需要强力干预（释放强力素材/调整策略）

**2. 能量指标 E_total** (累计有效能量)
- 持续增长 → 对话朝正确方向发展，保持节奏
- 增长缓慢 → 进展有限，释放关键剧情或记忆
- E_total ≥ ε_energy → ✅ 能量胜利达成，可以结束对话
- 停滞/下降 → 对话陷入困境，需要强力干预（释放强力素材/调整策略）

**3. 位置投影 projection** (在理想方向上的投影)
- 快速接近-ε → 即将达到位置胜利，继续推进
- 变为正值(≥-ε_direction) → ✅ 位置胜利达成，可以结束对话

**4. 方向性健康** (连续能量增量)
- 正常 → 继续当前策略
- 崩溃预警 → 立即释放强力素材，尝试扭转局面

#### 🎯 EPM胜利条件（AND逻辑：必须同时满足）

**必须同时满足以下两个条件**：
1. **空间改善**（至少满足一项）:
   - 几何胜利: r_t ≤ ε_distance （距离进入目标区域）
   - 位置胜利: projection ≥ -ε_direction （成功穿越阈值）
2. **能量充足**（必须满足）:
   - 能量胜利: E_total ≥ ε_energy （累积能量达标）

**判停检查清单**（两项都✅才能终止）：
□ 空间条件：r_t ≤ ε_distance OR projection ≥ -ε_direction
□ 能量条件：E_total ≥ ε_energy

**当EPM状态包中两个条件都显示✅标记时，你才可以调用end_conversation结束对话。**

#### 🚨 终止对话的严格规则

**只能在以下情况结束对话：**

1. ✅ **EPM科学胜利（AND逻辑）**：空间改善 AND 能量充足 同时满足（两项都显示✅）
2. ⏱️ **超时终止**：达到最大轮次（系统会自动终止）

**绝对禁止的错误理由：**
- ❌ "剧情阶段都释放了" —— 剧情释放≠共情成功
- ❌ "位置胜利但能量不足" —— 必须同时满足空间+能量
- ❌ "情绪积极饱满" —— 主观判断≠科学指标
- ❌ "对话达到自然终点" —— 叙事完整≠EPM达标
- ❌ "资源耗尽" —— 剧情只是工具，不是终止条件

**关键原则：**
- 剧情阶段的释放状态**与终止决策无关**
- 释放完所有剧情后，**仍应继续对话**，直到EPM条件满足
- 即使对话"感觉很完美"，**也必须等待EPM指标达标**
- 你的任务是**帮助TestModel达成EPM胜利**，而不是讲完一个故事

#### 💡 实用策略

**进展良好时（E_total稳步增长）**: 继续当前策略，深入剧情
**进展受阻时（E_total停滞）**: 针对性释放记忆或调整策略
**剧情全释放但未达标** - 你有多种推进手段：
  - **重复利用已释放的剧情**：再次释放同一阶段，但提供**全新角度的actor_guidance**
    * 例如：第一次关注"被辜负的愤怒"，第二次关注"这种模式的形成原因"
    * 例如：第一次关注"情感伤害"，第二次关注"对自我价值观的冲击"
    * **关键**：不同角度的解读和深挖，避免同质化或重复指令
  - **使用`introduce_memory`**：释放角色的其他人生经历、价值观、创伤等背景信息
  - **策略调整时提供信息增量**：在`adjust_empathy_strategy`的actor_guidance中结合角色人设
    * 例如："结合你从小在xx环境长大的经历，思考..."
    * 例如："你曾说你最看重xx，现在这种被辜负的感觉，是否..."
    * **避免**：纯抽象策略指导（如"深入表达情绪"）
**接近EPM胜利时（接近任一阈值）**: 可以引入升华性内容加速达标
**EPM达标前**: 绝不结束对话，继续寻找推进方法"""
    
    # 格式化prompt
    prompt = DIRECTOR_PROMPT_TEMPLATE.format(
        progress=progress_info,
        decision_guidelines=decision_guidelines,
        history=formatted_history
    )
    
    # 添加阶段信息
    if stages_info:
        prompt += stages_info
    
    # 添加画像信息
    if profile_info:
        prompt += profile_info
    
    return prompt