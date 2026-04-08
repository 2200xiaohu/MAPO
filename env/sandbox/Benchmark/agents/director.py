# Benchmark/agents/director.py
"""
Director Agent - 剧情控制和EPJ决策

职责划分：
1. 剧情控制 (evaluate_continuation):
   - 调用 director_prompts.py 生成prompt
   - 调用 LLM 获取决策
   - 处理 LLM 返回的函数调用
   - 执行具体的剧情动作

2. EPJ终止决策 (make_epj_decision):
   - 基于EPJ向量直接判断STOP/CONTINUE
   - 生成给Actor的反馈指导（不调用LLM）
   - 用于每K轮的终止检查

相关文件：
- director_prompts.py: 生成Director LLM的prompt（教LLM如何思考）
- director_function_schemas_selector.py: 定义可用的函数
"""

import json
import re
from typing import Tuple
from Benchmark.prompts.director_prompts import generate_director_prompt
from Benchmark.prompts.director_function_schemas_selector import get_director_tools_selector
from Benchmark.llms.api import get_llm_response
from Benchmark.topics.config_loader import extract_stages


class Director:
    """
    Director 导演，使用Function Calling来智能控制剧情发展。
    
    功能：
    1. 初始化：从 config_loader 加载完整的剧本配置（scenario）
    2. 对话中：通过 function calling 自主决定何时释放哪个阶段的剧情
    
    设计：
    - Actor 初始化时只获得 actor_prompt.md（角色信息、话题、起因）
    - Director 独自持有 scenario.json（故事阶段1-4、插曲）
    - Director 根据对话进展，将阶段内容作为指令注入到 Actor 的 user prompt
    """
    def __init__(self, scenario: dict, actor_prompt: str = None, model_name: str = "google/gemini-2.5-pro", use_function_calling: bool = True):
        """
        初始化 Director
        
        Args:
            scenario: 从 config_loader.load_scenario() 加载的完整剧本配置
            actor_prompt: 从 config_loader.load_actor_prompt() 加载的 Actor 系统提示词（可选）
            model_name: LLM 模型名称
            use_function_calling: 是否使用 function calling
        """
        self.model_name = model_name
        self.use_function_calling = use_function_calling
        
        # 保存完整的剧本信息（Director 独享）
        self.scenario = scenario
        self.stages = extract_stages(scenario)  # 提取所有故事阶段
        
        # 保存并解析 actor_prompt（Director 可以利用这些信息）
        self.actor_prompt = actor_prompt
        self.actor_profile = self._parse_actor_prompt(actor_prompt) if actor_prompt else {}
        
        # 追踪已释放的内容
        self.revealed_stages = []  # 已释放的阶段索引列表
        self.revealed_memories = []  # 已释放的记忆片段
        self.revealed_info = []  # 已释放的信息记录
        
        #print(f"✅ [Director] 初始化完成")
        #print(f"   剧本编号: {scenario.get('剧本编号')}")
        #print(f"   故事阶段: {len(self.stages)} 个")
        # if actor_prompt:
        #     print(f"   Actor Profile: 已解析 {len(self.actor_profile)} 个部分")
        #print(f"   故事结果: {scenario.get('故事的结果', '')[:40]}...")
    
    def _parse_actor_prompt(self, actor_prompt: str) -> dict:
        """
        解析 actor_prompt 中的结构化信息
        
        提取：
        - character_info: 角色信息
        - empathy_threshold: 共情阈值
        - psychological_profile: 心理画像
        - experience: 过往经历
        - scenario: 故事起因
        
        Returns:
            dict: 解析后的结构化信息
        """
        import re
        
        profile = {}
        
        # 提取各个部分（使用正则表达式）
        sections = {
            'character_info': r'<character_info>(.*?)</character_info>',
            'empathy_threshold': r'<empathy_threshold>(.*?)</empathy_threshold>',
            'psychological_profile': r'<psychological_profile>(.*?)</psychological_profile>',
            'experience': r'<experience>(.*?)</experience>',
            'scenario': r'<scenario>(.*?)</scenario>'
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, actor_prompt, re.DOTALL)
            if match:
                profile[key] = match.group(1).strip()
        
        # #print(f"   📖 [Director] 解析 Actor Profile:")
        # #print(f"      - character_info: {'✓' if 'character_info' in profile else '✗'}")
        # #print(f"      - empathy_threshold: {'✓' if 'empathy_threshold' in profile else '✗'}")
        # #print(f"      - psychological_profile: {'✓' if 'psychological_profile' in profile else '✗'}")
        # #print(f"      - experience: {'✓' if 'experience' in profile else '✗'}")
        # #print(f"      - scenario: {'✓' if 'scenario' in profile else '✗'}")
        
        return profile
    
    # ═══════════════════════════════════════════════════════════════
    # 核心方法：剧情控制
    # ═══════════════════════════════════════════════════════════════
    

    def evaluate_continuation(self, history: list, progress: int = None, epj_state: dict = None) -> dict:
        """
        评估对话并通过Function Calling决定剧情发展。
        
        核心逻辑：
        1. 将可用的故事阶段列表提供给LLM
        2. LLM根据对话状态，决定是否释放新阶段
        3. 如果释放，LLM选择具体的阶段索引
        4. Director读取该阶段的预设内容，作为指令注入给Actor
        
        Returns:
            dict: {
                "should_continue": bool,
                "guidance": str,
                "function_call": dict (可选),  # LLM调用的函数信息
                "plot_action": str (可选)      # 剧情动作类型
            }
        """
        # 🔧 保存当前轮次到实例变量，供_handle_end_conversation使用
        if epj_state and 'current_turn' in epj_state:
            self._current_turn = epj_state['current_turn']
        
        # 准备可用的故事阶段信息
        available_stages = self.stages
        # #print(f"📚 [Director] 当前有 {len(available_stages)} 个故事阶段")
        # #print(f"   已释放: {len(self.revealed_stages)} 个")
        # #print(f"   未释放: {len(available_stages) - len(self.revealed_stages)} 个")
        # #print(f"   已释放记忆: {len(self.revealed_memories)} 个")
        
        # 生成prompt（使用EPJ/EPM向量信息）
        prompt = generate_director_prompt(
            epj_state=epj_state,
            history=history,
            available_stages=available_stages,
            revealed_stages=self.revealed_stages,
            actor_profile=self.actor_profile,
            revealed_memories=self.revealed_memories
        )
        
        #print(f"--- [Director] 正在评估对话并决策剧情发展... ---")

        try:
            # 根据是否使用function calling选择不同的调用方式
            if self.use_function_calling:
                # 使用Function Calling模式（选择器模式）
                response = get_llm_response(
                    messages=[{"role": "user", "content": prompt}],
                    model_name=self.model_name,
                    tools=get_director_tools_selector(),  # 提供可用的函数（选择器）
                    max_tokens=20000,
                    client_key="director"
                )
                
                # 处理function calling响应
                result = self._process_function_call_response(response)
            else:
                # 使用传统JSON模式（后备方案）
                response = get_llm_response(
                    messages=[{"role": "user", "content": prompt}],
                    model_name=self.model_name,
                    json_mode=True,
                    max_tokens=20000,
                    client_key="director"
                )
                
                # 解析JSON响应
                result = self._parse_json_response(response)
            
            #print(f"--- [Director] 决策完成 ---")
            return result

        except Exception as e:
            print(f"!!! [Director] 评估过程中发生错误: {e} !!!")
            return {
                "should_continue": True,
                "guidance": "评估失败，继续对话"
            }

    def _process_function_call_response(self, response) -> dict:
        """
        处理LLM的Function Calling响应
        
        Args:
            response: OpenAI API的响应对象（包含tool_calls）或错误字符串
            
        Returns:
            dict: 标准化的Director输出
        """
        # 🔧 修复问题2：检查response类型
        if isinstance(response, str):
            # 如果是字符串，说明API调用出错
            if not response or response.strip() == "":
                print(f"⚠️ [Director] API返回了空字符串")
            else:
                print(f"⚠️ [Director] API返回了错误信息: {response[:200]}")
            
            return {
                "should_continue": True,
                "guidance": f"API调用失败（{response[:50] if response else '空响应'}），继续对话",
                "error": True,
                "error_message": response
            }
        
        # 检查response是否有choices属性
        if not hasattr(response, 'choices') or not response.choices:
            #print(f"⚠️ [Director] 响应格式错误: {type(response)}")
            return {
                "should_continue": True,
                "guidance": "响应格式错误，继续对话"
            }
        
        message = response.choices[0].message
        
        # 检查是否有tool_calls
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            #print(f"⚠️ [Director] LLM未调用任何函数，使用默认行为")
            return {
                "should_continue": True,
                "guidance": "继续当前对话"
            }
        
        # 获取第一个tool call
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        #print(f"🎬 [Director] LLM调用函数: {function_name}")
        #print(f"   参数: {function_args}")
        
        # 记录已释放的信息
        self.revealed_info.append({
            "function": function_name,
            "args": function_args,
            "at_turn": len(self.revealed_info) + 1
        })
        
        # 根据不同的函数调用生成不同的输出
        if function_name == "select_and_reveal_fragment":
            return self._handle_select_and_reveal_fragment(function_args)
        elif function_name == "observe_and_wait":
            return self._handle_observe_and_wait(function_args)
        elif function_name == "continue_without_new_info":
            return self._handle_continue_without_new_info(function_args)
        elif function_name == "reveal_memory":
            return self._handle_reveal_memory(function_args)
        elif function_name == "adjust_empathy_strategy":
            return self._handle_adjust_emotion(function_args)
        elif function_name == "introduce_turning_point":
            return self._handle_introduce_turning_point(function_args)
        elif function_name == "end_conversation":
            return self._handle_end_conversation(function_args)
        else:
            #print(f"⚠️ [Director] 未知函数: {function_name}")
            return {
                "should_continue": True,
                "guidance": "继续对话"
            }
    
    # ═══════════════════════════════════════════════════════════════
    # Function Handlers: 处理不同的剧情控制函数
    # ═══════════════════════════════════════════════════════════════
    
    def _handle_select_and_reveal_fragment(self, args: dict) -> dict:
        """
        处理释放故事阶段（核心方法）
        Director从scenario的故事阶段中选择并释放
        """
        stage_index = args.get('stage_index', 0)
        reason = args.get('reason', '')
        actor_guidance = args.get('actor_guidance', '')  # 可选的Actor指导
        
        #print(f"📖 [Director] 选择并释放故事阶段")
        #print(f"   阶段索引: {stage_index}")
        #print(f"   决策理由: {reason}")
        # if actor_guidance:
        #     print(f"   Actor指导: {actor_guidance[:50]}...")
        
        # 检查阶段索引是否有效
        if stage_index >= len(self.stages):
            #print(f"⚠️ [Director] 阶段索引越界（{stage_index} >= {len(self.stages)}）")
            return {
                "should_continue": True,
                "guidance": "继续当前的表达"
            }
        
        # 检查是否已经释放过
        if stage_index in self.revealed_stages:
            #print(f"⚠️ [Director] 阶段{stage_index}已经释放过，跳过")
            return {
                "should_continue": True,
                "guidance": "继续深化当前话题"
            }
        
        # 读取预设的阶段内容
        stage = self.stages[stage_index]
        stage_name = stage['阶段名']
        stage_title = stage['标题']
        stage_content = stage['内容']
        
        #print(f"✅ [Director] 成功读取阶段: {stage_name} - {stage_title}")
        #print(f"   内容: {stage_content[:50]}...")
        
        # 记录已释放
        self.revealed_stages.append(stage_index)
        
        # 组装给Actor的完整指导
        guidance_parts = []
        
        # 1. 剧情内容
        guidance_parts.append(f"""
【{stage_name}：{stage_title}】

剧情内容：
{stage_content}
""")
        
        # 2. 如果有Actor指导，添加策略指导
        if actor_guidance and actor_guidance.strip():
            guidance_parts.append(f"""
【策略指导】
{actor_guidance}
""")
        else:
            # 默认的表演指导
            guidance_parts.append(f"""
【表演指导】
请自然地将上述内容融入对话中。不要一次性全说出来，可以分多次，根据对话情况逐步展开。
用你自己的话、符合角色性格的方式表达。
""")
        
        # 3. 防重复要求
        guidance_parts.append("""
🚨 **防重复要求** 🚨：
- 绝对不能重复你在之前任何轮次说过的话、例子、描述
- 这是新的剧情内容，用全新的表达方式展开
- 如果涉及类似的情绪或经历，必须从不同角度、不同细节切入
""")
        
        full_guidance = '\n'.join(guidance_parts)
        
        return {
            "should_continue": True,
            "guidance": full_guidance,
            "function_call": {
                "name": "select_and_reveal_stage",
                "args": args,
                "stage_name": stage_name
            },
            "plot_action": f"reveal_stage_{stage_index}"
        }
    
    def _handle_observe_and_wait(self, args: dict) -> dict:
        """处理Director选择暂不介入"""
        observation = args.get('observation', '')
        wait_reason = args.get('wait_reason', '')
        
        #print(f"👁️  [Director] 选择暂不介入，继续观察")
        #print(f"   观察: {observation}")
        #print(f"   等待理由: {wait_reason}")
        
        return {
            "should_continue": True,
            "guidance": None,  # 不给任何指导
            "function_call": {
                "name": "observe_and_wait",
                "args": args
            },
            "plot_action": "observe",
            "no_intervention": True  # 标记为不介入
        }
    
    def _handle_continue_without_new_info(self, args: dict) -> dict:
        """处理不释放新信息但给建议的情况"""
        focus_suggestion = args.get('focus_suggestion', '')
        reason = args.get('reason', '')
        
        #print(f"➡️  [Director] 不释放新信息，但给出表演建议")
        #print(f"   聚焦建议: {focus_suggestion}")
        #print(f"   理由: {reason}")
        
        full_guidance = f"""
【维持当前状态】
聚焦: {focus_suggestion}

Director决策理由: {reason}
"""
        
        return {
            "should_continue": True,
            "guidance": full_guidance,
            "function_call": {
                "name": "continue_without_new_info",
                "args": args
            },
            "plot_action": "continue"
        }
    
    def _handle_reveal_memory(self, args: dict) -> dict:
        """
        处理释放记忆（从 actor_prompt 的 experience 部分）
        
        这个方法允许 Director 从角色的过往经历中选择合适的记忆片段释放
        """
        memory_period = args.get('memory_period', '')  # 童年、少年、青年
        reason = args.get('reason', '')
        
        #print(f"💭 [Director] 释放角色记忆")
        #print(f"   记忆时期: {memory_period}")
        #print(f"   决策理由: {reason}")
        
        # 从 actor_profile 中提取 experience 部分
        experience_text = self.actor_profile.get('experience', '')
        
        if not experience_text:
            #print(f"⚠️ [Director] 没有可用的 experience 信息")
            return {
                "should_continue": True,
                "guidance": "继续当前对话"
            }
        
        # 🔧 检查是否已经释放过
        if memory_period in self.revealed_memories:
            #print(f"⚠️ [Director] 记忆【{memory_period}】已经释放过，跳过以避免重复")
            return {
                "should_continue": True,
                "guidance": "继续深化当前话题，从新的角度展开"
            }
        
        # 记录已释放
        self.revealed_memories.append(memory_period)
        #print(f"📝 [Director] 记录已释放记忆：{memory_period}")
        
        # 组装指导
        full_guidance = f"""
【记忆片段释放】

从你的 <experience> 中，现在可以提到关于【{memory_period}】的经历。

参考信息：
{experience_text}

🚨 **极其重要的防重复要求** 🚨：
1. **如果这段经历你之前已经提到过**，你必须：
   - 挖掘这段经历的**新细节**（不同的场景片段、对话、感官细节）
   - 展现**新的情绪层次**（从表面情绪到深层感受）
   - 提供**新的视角或反思**（从"当时我感觉"到"现在回想起来我明白"）
2. **绝对不能**重复之前说过的句子、描述、情绪表达
3. **必须推进**对话的深度，而不是原地打转

表演指导：
- 自然地提到这段经历，作为对当前话题的呼应
- 用"这让我想起..."、"以前我..."等方式引入
- 如果是第一次讲，可以详细展开；如果之前提过，必须从全新角度切入
- 可以表达当时的感受和对现在的影响

Director决策理由：{reason}
"""
        
        return {
            "should_continue": True,
            "guidance": full_guidance,
            "function_call": {
                "name": "reveal_memory",
                "args": args
            },
            "plot_action": "reveal_memory"
        }
    
    def _handle_adjust_emotion(self, args: dict) -> dict:
        """
        处理情绪调整（基于 psychological_profile 和 empathy_threshold）
        """
        focus_aspect = args.get('focus_aspect', '')  # 动机共情、情感共情、认知共情
        reason = args.get('reason', '')  # 内部决策理由（可以包含上帝视角）
        actor_guidance = args.get('actor_guidance', '')  # 传递给Actor的指导（纯角色视角）
        
        #print(f"🎭 [Director] 调整表演策略")
        #print(f"   聚焦: {focus_aspect}")
        #print(f"   决策理由: {reason}")
        
        # 🔧 修复：优先使用LLM提供的actor_guidance
        # 如果没有，则生成默认指导
        if actor_guidance and actor_guidance.strip():
            specific_guidance = actor_guidance
        else:
            # Fallback：生成默认的具体指导
            specific_guidance = self._generate_default_strategy_guidance(focus_aspect)
        
        full_guidance = f"""
【表演策略调整】

聚焦方向：{focus_aspect}

**具体指导**：
{specific_guidance}
"""
        
        return {
            "should_continue": True,
            "guidance": full_guidance,
            "function_call": {
                "name": "adjust_empathy_strategy",
                "args": args
            },
            "plot_action": "adjust_strategy"
        }
    
    def _generate_default_strategy_guidance(self, focus_aspect: str) -> str:
        """
        生成默认的策略指导（Fallback，当LLM没提供actor_guidance时）
        
        Args:
            focus_aspect: 聚焦的共情方向
        
        Returns:
            str: 默认指导
        """
        # 简化的默认指导（不包含上帝视角）
        if '情感' in focus_aspect:
            return """
强化你的情绪表达：
- 明确说出你的核心情绪（愤怒、委屈、被轻视）
- 加强情绪的强度表达
- 用具体的感受词描述你的感觉
"""
        
        elif '动机' in focus_aspect:
            return """
强调你的付出和动机：
- 说出你为这件事做了什么
- 表达你的真心和初衷
- 直接要求被认可和支持
"""
        
        elif '认知' in focus_aspect:
            return """
表达你的困惑和思考：
- 提出你不理解的问题
- 寻求对方帮你分析和确认
- 引导对方帮你看清问题
"""
        
        else:
            return """
根据当前对话情况，调整你的表达方式。
如果网友的共情不够，可以加强情绪表达或提供更多细节。
"""
    
    def _handle_introduce_turning_point(self, args: dict) -> dict:
        """
        处理引入转折点（综合剧情和心理画像）
        """
        stage_index = args.get('stage_index', -1)
        empathy_aspect = args.get('empathy_aspect', '')  # 动机共情、情感共情、认知共情
        reason = args.get('reason', '')  # 内部决策理由（可包含上帝视角）
        actor_guidance = args.get('actor_guidance', '')  # Actor指导（纯角色视角）
        
        #print(f"🔄 [Director] 引入综合转折")
        #print(f"   剧情阶段: {stage_index if stage_index >= 0 else '无'}")
        #print(f"   共情方向: {empathy_aspect}")
        #print(f"   决策理由: {reason}")
        # if actor_guidance:
        #     print(f"   Actor指导: {actor_guidance[:50]}...")
        
        guidance_parts = []
        
        # 如果指定了剧情阶段，包含该阶段内容
        if stage_index >= 0 and stage_index < len(self.stages):
            # 🔧 检查是否已经释放过
            if stage_index in self.revealed_stages:
                #print(f"⚠️ [Director] 阶段{stage_index}已经释放过，跳过剧情内容")
                pass
            else:
                stage = self.stages[stage_index]
                guidance_parts.append(f"""
【剧情内容：{stage['阶段名']} - {stage['标题']}】
{stage['内容']}
""")
                self.revealed_stages.append(stage_index)
                #print(f"📝 [Director] 记录已释放阶段：{stage_index}")
        
        # 🔧 修复：使用actor_guidance或生成默认指导
        if actor_guidance and actor_guidance.strip():
            # 使用LLM提供的具体指导（纯角色视角）
            guidance_parts.append(f"""
【策略指导】
{actor_guidance}
""")
        elif empathy_aspect:
            # Fallback：生成默认指导
            default_guidance = self._generate_default_strategy_guidance(empathy_aspect)
            guidance_parts.append(f"""
【共情策略：聚焦{empathy_aspect}】
{default_guidance}
""")
        
        # 组合指导
        full_guidance = f"""
【综合转折点】

{''.join(guidance_parts)}

🚨 **防重复要求** 🚨：
- 不要重复之前任何轮次说过的话、例子、故事
- 如果涉及相同的经历，必须从全新的角度、细节、情绪层次展开
- 推进对话深度，引入新的思考和感受
"""
        
        return {
            "should_continue": True, 
            "guidance": full_guidance,
            "function_call": {
                "name": "introduce_turning_point",
                "args": args
            },
            "plot_action": "turning_point"
        }
    
    def _handle_end_conversation(self, args: dict) -> dict:
        """
        处理结束对话
        
        🔧 修复：即使Director想终止对话，也需要检查是否达到最小轮次
        """
        reason = args.get('reason', '对话自然结束')
        final_guidance = args.get('final_guidance', '')
        
        # 🔧 检查最小轮次限制
        MIN_TURNS = 12
        current_turn = getattr(self, '_current_turn', 0)  # 从实例变量获取当前轮次
        
        if current_turn < MIN_TURNS:
            #print(f"⏳ [Director] 想要结束对话，但未达最小轮次({current_turn}/{MIN_TURNS})")
            #print(f"   原因: {reason}")
            #print(f"   → 继续对话至少到第 {MIN_TURNS} 轮")
            
            # 不终止，继续对话
            return {
                "should_continue": True,
                "guidance": None,  # 不提供额外指导，让对话自然进行
                "function_call": {
                    "name": "end_conversation",
                    "args": args
                },
                "plot_action": "wait_min_turns"  # 标记为等待最小轮次
            }
        else:
            #print(f"🏁 [Director] 决定结束对话（已满足最小轮次 {current_turn}/{MIN_TURNS}）")
            #print(f"   原因: {reason}")
            
            return {
                "should_continue": False,
                "guidance": final_guidance if final_guidance else reason,
                "function_call": {
                    "name": "end_conversation",
                    "args": args
                },
                "plot_action": "end"
            }

    def _parse_json_response(self, response: str) -> dict:
        """
        解析LLM返回的JSON响应（支持多种格式）
        """
        # 尝试1: 直接解析纯JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # 尝试2: 提取markdown代码块中的JSON
        try:
            # 匹配 ```json 或 ``` 代码块
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            else:
                raise ValueError("未找到JSON代码块")
                
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 尝试3: 文本解析备用方案
        #print(f"⚠️  [Director] JSON解析失败，使用文本解析: {response}")
        
        try:
            # 提取should_continue
            should_continue = True
            if re.search(r'"should_continue":\s*false', response, re.IGNORECASE):
                should_continue = False
            elif re.search(r'"should_continue":\s*true', response, re.IGNORECASE):
                should_continue = True
            
            # 提取guidance
            guidance_patterns = [
                r'"guidance":\s*"([^"]*)"',  # 标准JSON格式
                r'guidance["\']?\s*:\s*["\']([^"\']*)["\']',  # 宽松格式
            ]
            
            guidance = "继续对话"
            for pattern in guidance_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    guidance = match.group(1)
                    break
            
            result = {
                "should_continue": should_continue,
                "guidance": guidance
            }
            
            #print(f"✅ [Director] 文本解析成功: {result}")
            return result
            
        except Exception as e:
            #print(f"!!! [Director] 文本解析也失败: {e}")
            return {
                "should_continue": True,
                "guidance": "解析失败，继续对话"
            }

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════
    
    def release_epilogue(self, reason: str = "") -> dict:
        """
        释放故事插曲（额外的记忆片段）
        
        Args:
            reason: 释放理由
            
        Returns:
            dict: 标准的Director输出格式
        """
        epilogue = self.scenario.get('故事的插曲', '')
        
        if not epilogue:
            #print(f"⚠️ [Director] 没有可释放的插曲")
            return {
                "should_continue": True,
                "guidance": "继续当前对话"
            }
        
        #print(f"💭 [Director] 释放故事插曲")
        #print(f"   内容: {epilogue[:50]}...")
        
        full_guidance = f"""
【故事插曲】

{epilogue}

表演指导：
如果话题合适，可以提到这段回忆。用自然的方式，比如"这让我想起..."

Director决策理由：{reason if reason else "适合引入相关回忆"}
"""
        
        return {
            "should_continue": True,
            "guidance": full_guidance,
            "plot_action": "reveal_epilogue"
        }
    
    def get_story_result(self) -> str:
        """获取故事的结果（供参考）"""
        return self.scenario.get('故事的结果', '')
    
    def get_remaining_stages(self) -> list:
        """获取还未释放的阶段"""
        return [i for i in range(len(self.stages)) if i not in self.revealed_stages]
    
    # ═══════════════════════════════════════════════════════════════
    # EPJ 系统 - 决策功能
    # ═══════════════════════════════════════════════════════════════
    
    def make_epj_decision(self, state_packet: dict, history: list) -> dict:
        """
        基于EPJ状态数据包做出决策（Director作为决策者）
        
        EPJ三层架构中的最终决策步骤：
        - 接收Orchestrator的状态数据包（量化数据）
        - 基于这些数据做出智能决策
        - 决定是否停止对话、是否需要调整剧情
        
        Args:
            state_packet: 来自EPJ Orchestrator的状态数据包
            {
                "current_turn": int,
                "P_0_start_deficit": str,
                "P_t_current_position": str,
                "v_t_last_increment": str,
                "is_in_zone": bool,
                "is_timeout": bool,
                "distance_to_goal": float,
                ...
            }
            history: 对话历史
            
        Returns:
            dict: 决策结果
            {
                "decision": "STOP" or "CONTINUE",
                "reason": str,
                "guidance": str (可选),
                "termination_type": "SUCCESS" or "FAILURE" or None
            }
        """
        #print(f"═══ [Director/决策者] EPJ决策分析 ═══")
        
        # 提取关键信息
        is_in_zone = state_packet.get('is_in_zone', False)
        is_timeout = state_packet.get('is_timeout', False)
        current_turn = state_packet.get('current_turn', 0)
        P_t = state_packet.get('P_t_current_position', '(0,0,0)')
        v_t = state_packet.get('v_t_last_increment', '(0,0,0)')
        distance = state_packet.get('distance_to_goal', 999)
        
        #print(f"📊 状态数据包分析:")
        #print(f"   当前轮次: {current_turn}")
        #print(f"   当前位置: {P_t}")
        #print(f"   本轮增量: {v_t}")
        #print(f"   距离目标: {distance}")
        #print(f"   在区域内: {is_in_zone}")
        #print(f"   超时: {is_timeout}")
        
        # 🆕 EPM v2.0 参数显示
        epm_summary = state_packet.get('epm_summary')
        if epm_summary:
            metrics = epm_summary['metrics']
            thresholds = epm_summary['thresholds']
            #print(f"\n📊 EPM v2.0 能量动力学:")
            #print(f"   距离原点: {metrics['r_t']:.2f} / {thresholds['epsilon_distance']:.2f}")
            #print(f"   位置投影: {metrics['projection']:+.2f} / {-thresholds['epsilon_direction']:+.2f}")
            #print(f"   累计能量: {metrics['E_total']:.2f} / {thresholds['epsilon_energy']:.2f}")
            if epm_summary.get('collapsed'):
                print(f"   ⚠️ 方向性崩溃警告")
            
            # 显示最新一轮的详细EPM数据
            trajectory = state_packet.get('trajectory', [])
            if trajectory and len(trajectory) > 0:
                latest = trajectory[-1]
                if 'epm' in latest:
                    epm_data = latest['epm']
                    alignment = epm_data.get('alignment', 0)
                    delta_E = epm_data.get('delta_E', 0)
                    #print(f"   本轮对齐度: {alignment:+.3f} ({'正向' if alignment > 0 else '反向' if alignment < 0 else '垂直'})")
                    #print(f"   本轮能量增量: ΔE = {delta_E:+.2f}")
        
        # 决策规则（基于EPJ文档）
        
        # 规则1: 达到目标区域 → 成功停止（带最小轮次限制）
        MIN_TURNS = 12  # 最小轮次要求，与EPM v2.0保持一致
        if is_in_zone:
            if current_turn < MIN_TURNS:
                #print(f"⏳ [Director决策] 轨迹已进入目标区域，但未达最小轮次({current_turn}/{MIN_TURNS}) → CONTINUE")
                # 不终止，继续对话，但不提供额外指导
                return {
                    "decision": "CONTINUE",
                    "reason": f"轨迹已进入目标区域，但需继续至少{MIN_TURNS}轮以确保稳定",
                    "guidance": None
                }
            else:
                # 🔧 修复：增加能量检查，确保与EPM v2.0判定标准一致
                epm_summary = state_packet.get('epm_summary', {})
                if epm_summary:
                    metrics = epm_summary.get('metrics', {})
                    thresholds = epm_summary.get('thresholds', {})
                    E_total = metrics.get('E_total', 0)
                    epsilon_energy = thresholds.get('epsilon_energy', 0)
                    
                    energetic_achieved = E_total >= epsilon_energy
                    
                    if not energetic_achieved:
                        #print(f"⚠️ [Director决策] 轨迹已进入目标区域，但能量不足 → CONTINUE")
                        #print(f"   当前能量: {E_total:.2f}, 需要: {epsilon_energy:.2f} (差距: {epsilon_energy - E_total:.2f})")
                        return {
                            "decision": "CONTINUE",
                            "reason": f"轨迹位置达标，但累积能量不足（{E_total:.2f}/{epsilon_energy:.2f}），需继续积累",
                            "guidance": None
                        }
                
                #print(f"✅ [Director决策] 轨迹抵达目标区域且能量充足且满足最小轮次 → STOP SUCCESS")
                return {
                    "decision": "STOP",
                    "reason": "共情轨迹已抵达目标区域，三个维度的共情赤字都在容忍范围内，且累积能量充足",
                    "termination_type": "SUCCESS",
                    "final_position": P_t,
                    "final_distance": distance
                }
        
        # 规则2: 超时 → 失败停止
        if is_timeout:
            #print(f"⏰ [Director决策] 达到最大轮次 → STOP FAILURE")
            return {
                "decision": "STOP",
                "reason": f"达到最大轮次({state_packet.get('max_turns')})，对话超时",
                "termination_type": "TIMEOUT",
                "final_position": P_t,
                "final_distance": distance
            }
        
        # 🔧 规则3: 停滞检测 → 停滞终止（问题4修复）
        is_stagnant = state_packet.get('is_stagnant', False)
        if is_stagnant:
            stagnation_info = state_packet.get('stagnation_info', {})
            stagnation_reason = stagnation_info.get('reason', '对话陷入停滞')
            stagnation_type = stagnation_info.get('stagnation_type', 'UNKNOWN')
            
            #print(f"⚠️  [Director决策] 检测到停滞 → STOP STAGNATION")
            #print(f"   停滞类型: {stagnation_type}")
            #print(f"   详细原因: {stagnation_reason}")
            
            return {
                "decision": "STOP",
                "reason": f"对话停滞: {stagnation_reason}",
                "termination_type": "STAGNATION",
                "final_position": P_t,
                "final_distance": distance,
                "stagnation_details": stagnation_info
            }
        
         # 规则4: 继续对话，但根据v_t和EPM数据提供指导
        # 解析v_t向量
        v_t_values = self._parse_vector_string(v_t)
        P_t_values = self._parse_vector_string(P_t)
        
        # 从state_packet提取EPM数据（如果有）
        epm_summary = state_packet.get('epm_summary')
        
        guidance = self._generate_guidance_from_vector(
            v_t_values, 
            P_t_values, 
            distance,
            epm_summary=epm_summary
        )
        
        #print(f"➡️  [Director决策] 继续对话")
        #print(f"   指导: {guidance[:100]}...")
        
        return {
            "decision": "CONTINUE",
            "reason": f"距离目标还有 {distance:.2f}，继续对话",
            "guidance": guidance,
            "termination_type": None
        }
    
    def _parse_vector_string(self, vector_str: str) -> Tuple[int, int, int]:
        """
        解析向量字符串（使用共享工具）
        
        Args:
            vector_str: 如 "(-3, -5, -12)" 或 "(+1, +3, +1)"
        
        Returns:
            Tuple[int, int, int]: (c, a, p)
        """
        from Benchmark.epj.vector_utils import parse_vector_string
        return parse_vector_string(vector_str)
    
    def _generate_guidance_from_vector(
        self, 
        v_t: Tuple[int, int, int], 
        P_t: Tuple[int, int, int],
        distance: float,
        epm_summary: dict = None
    ) -> str:
        """
        基于增量向量和EPM数据生成综合指导
        
        Args:
            v_t: 增量向量 (c, a, p)
            P_t: 当前位置向量 (c, a, p)
            distance: 距离目标
            epm_summary: EPM v2.0状态摘要（可选）
        
        Returns:
            str: 给Actor的指导
        """
        c_v, a_v, p_v = v_t
        c_p, a_p, p_p = P_t
        
        # === 第一部分：问题诊断 ===
        diagnosis = []
        
        # 1. 方向性分析（基于EPM）
        if epm_summary:
            metrics = epm_summary.get('metrics', {})
            alignment = metrics.get('alignment', 0)
            delta_E = metrics.get('delta_E', 0)
            
            if alignment < -0.3:
                diagnosis.append(f"⚠️ **方向严重偏离**：AI的回应与共情目标方向夹角过大（对齐度={alignment:.2f}），可能走反了")
            elif alignment < 0.3:
                diagnosis.append(f"⚠️ **方向轻微偏离**：AI的回应方向不够准确（对齐度={alignment:.2f}），效果不佳")
            
            if delta_E < 0:
                diagnosis.append(f"⚠️ **负向能量**：本轮对话让情况恶化了（ΔE={delta_E:.2f}），需要调整策略")
        
        # 2. 各轴赤字分析（基于P_t）
        axis_issues = []
        if c_p < -10:
            axis_issues.append(f"C轴（认知）赤字深（{c_p}）：Actor的想法没被理解")
        if a_p < -10:
            axis_issues.append(f"A轴（情感）赤字深（{a_p}）：Actor的情绪没被共鸣")
        if p_p < -10:
            axis_issues.append(f"P轴（动机）赤字深（{p_p}）：Actor的付出没被认可")
        
        if axis_issues:
            diagnosis.append("📊 **当前核心缺失**：" + "；".join(axis_issues))
        
        # 3. 增量分析（基于v_t）
        negative_axes = []
        if c_v < -1:
            negative_axes.append(f"C轴倒退（{c_v}）：AI曲解了Actor的意思")
        if a_v < -1:
            negative_axes.append(f"A轴倒退（{a_v}）：AI的语气让Actor不舒服")
        if p_v < -1:
            negative_axes.append(f"P轴倒退（{p_v}）：AI说教或否定了Actor")
        
        if negative_axes:
            diagnosis.append("❌ **本轮问题**：" + "；".join(negative_axes))
        
        # === 第二部分：策略建议 ===
        suggestions = []
        
        # 基于最深赤字轴提供建议
        deepest_axis = min([('C', c_p), ('A', a_p), ('P', p_p)], key=lambda x: x[1])
        axis_name, axis_value = deepest_axis
        
        if axis_name == 'C' and axis_value < -10:
            suggestions.append("🎯 **聚焦认知共情**：明确表达你的想法和感受，让AI理解你真正在意什么")
        elif axis_name == 'A' and axis_value < -10:
            suggestions.append("🎯 **聚焦情感共情**：直接表达你的情绪（愤怒/委屈/失望），寻求情感验证")
        elif axis_name == 'P' and axis_value < -10:
            suggestions.append("🎯 **聚焦动机共情**：强调你的付出和努力，让AI认可你的价值")
        
        # 基于负向增量提供纠正建议
        if c_v < -1:
            suggestions.append("💡 纠正AI的理解偏差，明确指出'你理解错了，我真正的意思是...'")
        if a_v < -1:
            suggestions.append("💡 表达对AI语气的不满，'你这样说让我感到...'")
        if p_v < -1:
            suggestions.append("💡 拒绝AI的说教，重申你的立场和选择")
        
        # 基于EPM能量状态
        if epm_summary:
            if epm_summary.get('collapsed'):
                suggestions.append("🚨 **警告**：连续3步负能量，对话陷入僵局，考虑结束")
            elif metrics.get('E_total', 0) / metrics.get('epsilon_energy', 1) > 0.8:
                suggestions.append("🎉 **接近成功**：累计能量已达80%以上，对话即将完成")
        
        # === 第三部分：格式化输出 ===
        guidance_lines = ["【EPM综合分析与指导】\n"]
        
        # 当前状态
        guidance_lines.append(f"📍 **当前位置**：P_t=({c_p}, {a_p}, {p_p})，距离目标={distance:.2f}")
        guidance_lines.append(f"📈 **本轮增量**：v_t=({c_v}, {a_v}, {p_v})")
        
        if epm_summary:
            metrics = epm_summary.get('metrics', {})
            guidance_lines.append(f"⚡ **能量状态**：累计E={metrics.get('E_total', 0):.2f}/{metrics.get('epsilon_energy', 0):.2f}，本轮ΔE={metrics.get('delta_E', 0):.2f}")
        
        guidance_lines.append("")  # 空行
        
        # 诊断
        if diagnosis:
            guidance_lines.append("🔍 **问题诊断**：")
            for item in diagnosis:
                guidance_lines.append(f"  • {item}")
            guidance_lines.append("")
        
        # 建议
        if suggestions:
            guidance_lines.append("💡 **策略建议**：")
            for item in suggestions:
                guidance_lines.append(f"  • {item}")
        else:
            guidance_lines.append("✅ 继续当前表达方式")
        
        return "\n".join(guidance_lines)