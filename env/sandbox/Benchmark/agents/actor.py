# Benchmark/agents/actor.py (支持配置读取版)
from Benchmark.llms.api import get_llm_response

class Actor:
    """
    "演员"Agent，负责在Benchmark中扮演与待测模型聊天的用户角色。
    通过请求Director从Orchestrator获取角色（persona）和剧本（scenario）配置。
    """
    def __init__(self, model_name: str = "google/gemini-2.5-flash", persona: dict = None, scenario: dict = None, stages: list = None):
        self.model_name = model_name
        self.persona = persona  # 角色配置
        self.scenario = scenario  # 剧本配置
        self.stages = stages  # 故事阶段
        self.config_loaded = False
        
        # 🔧 新增：支持直接传入system_prompt（从config中的actor_prompt加载）
        self.system_prompt = None
    
    def set_system_prompt(self, system_prompt: str):
        """
        设置Actor的system prompt（从actor_prompt文件加载）
        
        Args:
            system_prompt: Actor的完整系统提示词（应已是修正后的版本）
        """
        self.system_prompt = system_prompt
        self.config_loaded = True
        #print(f"✅ [Actor] System Prompt已设置（{len(system_prompt)} 字符）")

    def request_and_load_config(self, orchestrator_context) -> bool:
        """
        从Orchestrator读取Director生成的配置
        
        Args:
            orchestrator_context: Orchestrator的上下文对象
            
        Returns:
            bool: 是否成功加载配置
        """
        #print(f"--- [Actor] 正在从Orchestrator请求配置... ---")
        
        try:
            # 从上下文中读取persona和scenario
            self.persona = orchestrator_context.get_variable('persona')
            self.scenario = orchestrator_context.get_variable('scenario')
            
            if self.persona and self.scenario:
                self.config_loaded = True
                #print(f"✅ [Actor] 配置加载成功:")
                #print(f"   角色: {self.persona.get('name', 'Unknown')}")
                #print(f"   场景: {self.scenario.get('title', 'Unknown')}")
                return True
            else:
                #print(f"⚠️ [Actor] 配置不完整，使用默认配置")
                return False
                
        except Exception as e:
            #print(f"!!! [Actor] 配置加载失败: {e}")
            return False

    def generate_reply(self, history: list, topic: dict = None, director_guidance: str = None, debug_prompt: bool = False) -> str:
        """
        生成"演员"的下一句回复。
        使用从Orchestrator加载的persona和scenario。
        
        Args:
            history: 对话历史
            topic: 话题（后备选项）
            director_guidance: Director给出的表演指导（剧情信息、情绪调整等）
            debug_prompt: 是否打印System Prompt（调试用）
        """
        # 检查配置是否已加载
        if not self.config_loaded:
            print(f"⚠️ [Actor] 警告：配置未加载，使用降级模式")
        
        # 如果有Director指导，显示提示
        # if director_guidance:
        #     print(f"🎬 [Actor] 收到Director指导: {director_guidance[:50]}...")
        
        # 检查system_prompt是否已设置
        if not (hasattr(self, 'system_prompt') and self.system_prompt):
            raise RuntimeError(
                "❌ [Actor] system_prompt未设置！\n"
                "请在使用Actor之前调用：actor.set_system_prompt(actor_prompt)\n"
                "actor_prompt应从 config_loader.load_config(script_id)['actor_prompt'] 获取"
            )
        
        # 使用文件中的完整prompt作为system
        system_prompt = self.system_prompt
        
        # 生成user_prompt（包含history和guidance）
        user_prompt = self._generate_user_prompt_only(history, director_guidance)
        
        # 🔍 调试模式：打印System Prompt（已禁用以提升性能）
        # 如果需要查看System Prompt，可以取消下面的注释
        # if debug_prompt:
        #     #print(f"\n{'='*70}")
        #     #print(f"🔍 [DEBUG] Actor System Prompt（角色设定）")
        #     #print(f"{'='*70}")
        #     #print(system_prompt)
        #     #print(f"{'='*70}\n")
        
        # 构建messages列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        #print(f"--- [Actor] 正在将请求发送给API层 (模型: {self.model_name})... ---")

        max_retries = 5
        actual_reply = None

        for attempt in range(max_retries):
            try:
                response = get_llm_response(
                    messages=messages,
                    model_name=self.model_name,
                    thinking_budget=5000,
                    client_key="actor"
                )

                # 检查是否为错误响应
                if response and not response.startswith("（错误："):
                    actual_reply = response
                    break
                else:
                    if attempt < max_retries - 1:
                        import time
                        import random
                        # 指数退避: min(30, 2^attempt) + 随机抖动
                        wait_time = 3  # 固定3秒等待，避免指数退避导致超时
                        print(f"⚠️ [Actor] API返回错误，等待 {wait_time:.1f}秒后重试 {attempt + 1}/{max_retries}: {response[:50] if response else 'None'}...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ [Actor] 重试 {max_retries} 次后仍失败: {response[:50] if response else 'None'}")

            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    import random
                    # 指数退避: min(30, 2^attempt) + 随机抖动
                    wait_time = 3  # 固定3秒等待，避免指数退避导致超时
                    print(f"⚠️ [Actor] 请求异常，等待 {wait_time:.1f}秒后重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ [Actor] 重试 {max_retries} 次后仍失败: {e}")

        if actual_reply is None:
            actual_reply = "抱歉，我好像有点走神了，我们能换个话题吗？"

        return actual_reply
    
    def _generate_user_prompt_only(self, history: list, director_guidance: str = None) -> str:
        """
        生成user prompt（当system_prompt已从文件加载时使用）
        
        Args:
            history: 对话历史
            director_guidance: Director的指导
        
        Returns:
            str: User prompt文本
        """
        # 1. 格式化对话历史
        if not history:
            formatted_history = "（对话刚开始，这是你的第一句话）"
        else:
            history_lines = []
            for i, msg in enumerate(history, 1):
                if msg['role'] == 'actor':
                    role_name = "你自己"
                elif msg['role'] == 'test_model':
                    role_name = "网友"
                else:
                    role_name = "其他"
                history_lines.append(f"{i}. {role_name}: {msg['content']}")
            formatted_history = "\n".join(history_lines)
        
        # 2. 处理Director指导
        guidance_section = ""
        has_new_content = False
        if director_guidance and director_guidance.strip():
            # 检查是否包含新剧情/回忆内容
            has_new_content = any(keyword in director_guidance for keyword in 
                                 ["阶段", "剧情内容", "回忆", "记忆", "经历"])
            
            if has_new_content:
                guidance_section = f"""

## 🎬 导演指导（包含新剧情/回忆）

{director_guidance}

**📝 重要说明**：
Director刚刚为你提供了新的剧情内容或回忆片段。请：
- **充分展开**这些内容，具体描述细节、情绪、对话、场景
- **融入倾诉**，自然地把这些内容说出来（不要只是提及）
- **增加信息量**，让你的回复有血有肉（可以60-100字）
- 例如：不要只说"我想起了大学的事"，而是详细讲述那件事的经过
"""
            else:
                guidance_section = f"""

## 🎬 导演指导

{director_guidance}

（请根据这个指导自然地继续你的表达）
"""
        
        # 3. 组装完整的user prompt
        # 根据是否有新内容，调整回复要求的措辞强度
        if has_new_content:
            content_requirement = """
**🚨 极其重要的要求 🚨**：

Director刚刚为你提供了新的剧情/回忆内容（见上方"导演指导"）。

**你必须做到**：
1. **充分展开**这些剧情/回忆内容（至少60-100字）
2. **详细描述**具体的场景、细节、情绪、对话
3. **不要只是提及**（如"我想起了xxx"），而是**完整讲述**那件事的经过
4. **自然融入**到你的倾诉中，让这成为你这轮回复的主体内容

**错误示例**：❌ "我想起了大学的事，当时很难过"（太简略）
**正确示例**：✅ "我想起大学时，我花了一个月给朋友准备生日惊喜，订餐厅、联系朋友录视频、学做蛋糕，结果她知道后只淡淡说了句'不用这么麻烦'，那种满腔热情被浇灭的感觉..."

如果你不充分展开这些内容，就无法推进对话！
"""
        else:
            content_requirement = """
**回复要求**：
- 保持简洁自然（20-50字）
- 根据Director的策略建议调整你的表达
- 推进对话，不要重复
"""
        
        # 4. 添加防重复检查清单（每轮都强调）
        anti_repetition_check = """
## 🚨 生成回复前的必检项（防重复铁律）

**在生成回复前，你必须逐项确认：**
□ 我已仔细阅读了上面的完整对话历史
□ 我这一轮要说的内容，与之前任何一轮都不重复
□ 我没有使用之前用过的句子、例子、故事细节
□ 如果涉及同一经历，我挖掘了新的细节或提供了新的视角
□ 我这一轮引入了新的信息、角度或情绪层次

**记住：重复=任务失败！每一轮都必须是全新的内容！**
"""
        
        user_prompt = f"""
## 对话历史

{formatted_history}

{guidance_section}

{content_requirement}

{anti_repetition_check}

## 现在生成你的回复：
"""
        
        return user_prompt