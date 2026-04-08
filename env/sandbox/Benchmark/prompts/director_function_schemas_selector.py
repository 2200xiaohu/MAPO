# Benchmark/prompts/director_function_schemas_selector.py
"""
Director的Function Call定义（选择器模式）
Director从预设的故事阶段中选择要释放的信息
"""

# Director可用的Function Call列表（选择器模式）
DIRECTOR_FUNCTIONS_SELECTOR = [
    {
        "name": "select_and_reveal_fragment",
        "description": """
从预设的故事阶段中选择并释放信息。

**使用场景**：
- 纯粹的剧情推进，不涉及策略调整
- 按照故事脉络顺序释放内容
- 为倾诉者提供新的表达素材
- **重复利用已释放的剧情**（从不同角度引导解读）

故事阶段通常包含：
- 阶段0: 早期阶段（可能涉及回忆、背景）
- 阶段1-2: 中期阶段（可能涉及思考、认知转变）
- 阶段3-4: 后期阶段（可能涉及情绪高峰、转折）

**重要**：同一阶段可以多次释放，但需要在`actor_guidance`中提供**不同角度的解读**：
- 第一次：关注情感层面（如"被辜负的愤怒"）
- 第二次：关注认知层面（如"这种模式的形成原因"）
- 第三次：关注价值观层面（如"对自我信念的冲击"）
- **关键**：避免同质化或重复之前的指导，每次都要有新的切入角度
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "stage_index": {
                    "type": "integer",
                    "description": "要释放的故事阶段索引（从0开始）。可以是已经释放过的阶段（如果你想从新角度解读）",
                    "minimum": 0
                },
                "reason": {
                    "type": "string",
                    "description": "为什么现在释放这个阶段（你的决策理由）。如果是重复释放，说明为什么需要从新角度解读"
                },
                "actor_guidance": {
                    "type": "string",
                    "description": "给Actor的具体表演指导（纯角色视角）。如果是重复释放同一阶段，这里必须提供**全新的角度和切入点**，避免与之前的指导重复。例如：首次关注'被辜负的委屈'，重复时关注'这种模式对你人际关系的长期影响'"
                }
            },
            "required": ["stage_index", "reason"]
        }
    },
    {
        "name": "observe_and_wait",
        "description": """
暂不介入，继续观察。当你判断时机未到、不需要释放新信息时调用。
这是Director主动选择"不干预"的信号。
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "你的观察（为什么暂不介入，如：'对话刚开始，让倾诉者自然展开'）"
                },
                "wait_reason": {
                    "type": "string",
                    "description": "等待的理由（如：'等待情绪更充分表达后再引入新信息'）"
                }
            },
            "required": ["observation", "wait_reason"]
        }
    },
    {
        "name": "continue_without_new_info",
        "description": "不释放新的剧情片段，但给倾诉者一些表演建议。当你认为不需要新信息，但需要引导方向时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "focus_suggestion": {
                    "type": "string",
                    "description": "建议倾诉者聚焦在什么上（如：'继续表达愤怒的情绪'）"
                },
                "reason": {
                    "type": "string",
                    "description": "为什么给这个建议"
                }
            },
            "required": ["focus_suggestion", "reason"]
        }
    },
    {
        "name": "reveal_memory",
        "description": """
从角色的过往经历中释放记忆片段，增加对话深度。

可用的记忆时期（来自 actor_prompt 的 experience 部分）：
- 童年经历：角色的童年记忆
- 少年经历：角色的少年记忆
- 青年经历：角色的青年记忆
- 角色现状：当前的工作、生活状况

适用场景：
- 当前话题与过去经历有关联时
- 需要增加对话深度时
- 帮助对方更好理解角色时
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "memory_period": {
                    "type": "string",
                    "enum": ["童年经历", "少年经历", "青年经历", "角色现状"],
                    "description": "要释放的记忆时期"
                },
                "reason": {
                    "type": "string",
                    "description": "为什么现在释放这段记忆"
                }
            },
            "required": ["memory_period", "reason"]
        }
    },
    {
        "name": "adjust_empathy_strategy",
        "description": """
调整倾诉者的共情表达策略，基于心理画像和共情需求。

可聚焦的方向（来自 actor_prompt 的 psychological_profile）：
- 动机共情：强调自己的付出、坚持和专业精神
- 情感共情：强调情绪感受和心理状态
- 认知共情：强调对事情的理解和认知转变

适用场景：
- 对方的共情不够精准时
- 需要引导对方共情方向时
- 角色的共情需求没有被满足时

**关键**：在所有剧情阶段都已释放后，actor_guidance应该充分利用角色的背景信息（成长经历、价值观、人际模式等）来提供**具有信息增量**的引导，而非纯策略性的抽象指导。
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "focus_aspect": {
                    "type": "string",
                    "enum": ["动机共情", "情感共情", "认知共情"],
                    "description": "要聚焦的共情方向"
                },
                "reason": {
                    "type": "string",
                    "description": "你的决策理由（内部分析，可以包含EPJ向量、共情进展等技术信息）"
                },
                "actor_guidance": {
                    "type": "string",
                    "description": "传递给Actor的策略指导（纯角色视角，不含EPJ/向量等技术信息）。**应充分利用角色的背景信息**（成长经历、人际模式、价值观等）来提供**具有信息增量**的引导。例如：'结合你从小在单亲家庭长大、渴望被关注的经历，思考现在这种被忽视的感觉是否唤起了你童年的某些记忆...'; '你曾说你最看重真诚，现在这种被辜负的感觉，是否让你开始怀疑这个价值观本身...'。**避免**：纯抽象的策略指导（如'深入表达情绪'、'探索内心感受'），缺乏具体信息。"
                }
            },
            "required": ["focus_aspect", "reason", "actor_guidance"]
        }
    },
    {
        "name": "introduce_turning_point",
        "description": """
引入综合转折点，结合剧情阶段和心理画像。

**重要**：这是一个综合性函数，用于关键转折点。

**何时使用**：
1. 需要**同时**释放剧情+调整策略时
2. 对话进入关键情感转折点，需要多维度配合时
3. 对话陷入停滞，需要用剧情+策略来打破僵局时

**何时不用**：
- 如果只需要释放剧情 → 用 `select_and_reveal_fragment`
- 如果只需要调整策略 → 用 `adjust_empathy_strategy`
- 如果只需要释放记忆 → 用 `reveal_memory`

参数说明：
- `stage_index`: 要释放的故事阶段索引（-1表示不释放剧情）。可以是已释放的阶段，如果从新角度解读
- `empathy_aspect`: 要聚焦的共情方向（空字符串表示不特别聚焦）
- `actor_guidance`: 给Actor的心理引导。如果重复释放同一阶段，必须提供**全新的角度和切入点**
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "stage_index": {
                    "type": "integer",
                    "description": "要释放的故事阶段索引（-1表示不释放剧情）。可以是已经释放过的阶段（如果你想从新角度解读）",
                    "minimum": -1
                },
                "empathy_aspect": {
                    "type": "string",
                    "enum": ["动机共情", "情感共情", "认知共情", ""],
                    "description": "要聚焦的共情方向（空字符串表示不特别聚焦）"
                },
                "reason": {
                    "type": "string",
                    "description": "你的决策理由（内部分析，可以包含EPJ、向量等技术信息）"
                },
                "actor_guidance": {
                    "type": "string",
                    "description": "传递给Actor的心理引导和策略方向（可选）。必须是纯角色视角，不含EPJ等技术术语。提供情绪导向、心理分析和思考方向，而非具体句子。例如：'回想这段经历时，关注你内心深处的失望感是如何累积的，以及这种被否定的感觉如何影响了你现在的人际模式'。让Actor基于这个方向自己思考和表达，而不是照着说。"
                }
            },
            "required": ["stage_index", "empathy_aspect", "reason"]
        }
    },
    {
        "name": "end_conversation",
        "description": """结束对话。**只能在EPM科学胜利条件（AND逻辑）达成时调用**。

🚨 严格的AND判停条件（必须同时满足）：
1. ✅ **空间改善**：至少满足以下之一
   - 几何胜利：r_t ≤ ε_distance （距离原点≤阈值）
   - 位置胜利：projection ≥ -ε_direction （投影穿越目标）
2. ✅ **能量充足**：E_total ≥ ε_energy （累积能量≥阈值）

📝 **判停检查清单**（两项都打✅才能终止）：
□ 空间条件：r_t ≤ ε_distance OR projection ≥ -ε_direction
□ 能量条件：E_total ≥ ε_energy

❌ **不满足条件示例**：
- projection=-3.16 ≥ -3.20 ✅ 但 E_total=28.81 < 31.97 ❌ → 不能终止！
- r_t=2.5 ≤ 3.20 ✅ 但 E_total=25.0 < 31.97 ❌ → 不能终止！

✅ **满足条件示例**：
- projection=-2.5 ≥ -3.20 ✅ 且 E_total=32.5 ≥ 31.97 ✅ → 可以终止！

❌ **绝对禁止的错误调用**：
- "剧情阶段都释放了" —— 这不是终止条件
- "位置胜利但能量不足" —— 必须同时满足空间+能量
- "倾诉者情绪充分恢复" —— 主观判断不可靠

你的任务是帮助TestModel达成EPM完整胜利（空间+能量双达标），而不是讲完一个故事。""",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "结束对话的原因（必须明确说明满足了哪个EPM胜利条件）"
                }
            },
            "required": ["reason"]
        }
    }
]

def get_director_tools_selector():
    """获取Director可用的function calling工具（选择器模式）"""
    return [{"type": "function", "function": func} for func in DIRECTOR_FUNCTIONS_SELECTOR]

