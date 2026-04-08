# Benchmark/orchestrator/chat_loop_epj.py
"""
基于EPJ系统的对话循环

EPJ三层架构：
1. Judger (传感器) - 填写心理量表
2. Orchestrator (计算器) - 计算向量和生成状态数据包
3. Director (决策者) - 基于状态数据包做决策
"""

from Benchmark.orchestrator.epj_orchestrator import EPJOrchestrator


def run_chat_loop_epj(actor, director, judger, test_model, script_id: str, max_turns=30, K=3, test_model_name: str = None):
    """
    基于EPJ系统的对话循环
    
    Args:
        actor: Actor实例
        director: Director实例
        judger: Judger实例
        test_model: TestModel实例
        script_id: 剧本ID（如"001"）
        max_turns: 最大轮次
        K: 评估周期（每K轮评估一次）
        test_model_name: 被测试模型名称（可选，用于结果记录）
    
    Returns:
        dict: 对话结果
    """
    from Benchmark.topics.config_loader import load_config
    
    print(f"\n{'='*70}")
    print(f"🎭 EPJ Benchmark 开始")
    print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化阶段
    # ═══════════════════════════════════════════════════════════════
    
    print(f"🎬 阶段1: 加载配置")
    
    # 加载剧本配置
    config = load_config(script_id)
    actor_prompt = config['actor_prompt']
    scenario = config['scenario']
    
    print(f"✅ 剧本配置加载完成")
    print(f"   剧本编号: {scenario.get('剧本编号')}")
    print(f"   Actor Prompt: {len(actor_prompt)} 字符")
    
    # 🔧 优化：提前提取Judger所需的精简上下文（只提取一次）
    from Benchmark.epj.judger_prompts import _extract_judger_context
    judger_context = _extract_judger_context(actor_prompt)
    print(f"   Judger Context: {len(judger_context)} 字符 (压缩率: {len(judger_context)/len(actor_prompt)*100:.1f}%)")
    
    # 初始化EPJ Orchestrator
    epj_orch = EPJOrchestrator(judger, threshold_type="high_threshold", K=K, max_turns=max_turns)
    
    # ═══════════════════════════════════════════════════════════════
    # T=0: EPJ初始化 - 使用预先计算的IEDR
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n🎬 阶段2: EPJ初始化 (T=0) - 加载预先计算的IEDR")
    
    # 🔧 优化：script_content传递精简的judger_context而不是完整actor_prompt
    script_content = {
        "actor_prompt": actor_prompt,  # 完整版（给IEDR初始化用，如果需要）
        "judger_context": judger_context,  # 精简版（给MDEP-PR评估用）
        "scenario": scenario
    }
    
    # 🔧 优化: 从批量评估结果中加载IEDR，避免重复计算
    from Benchmark.epj.iedr_loader import load_precomputed_iedr
    
    # 尝试加载预先计算的IEDR
    precomputed_data = load_precomputed_iedr(script_id)
    
    if precomputed_data and precomputed_data['status'] == 'success':
        # 使用预先计算的IEDR
        filled_iedr = precomputed_data['iedr']
        P_0_dict = precomputed_data['P_0']
        P_0 = (P_0_dict['C'], P_0_dict['A'], P_0_dict['P'])
        
        # 🆕 附加EPM参数（如果存在）
        if 'epm' in precomputed_data:
            filled_iedr['epm'] = precomputed_data['epm']
        
        print(f"✅ 从批量评估结果加载IEDR")
        print(f"   IEDR: C=[{filled_iedr['C.1']},{filled_iedr['C.2']},{filled_iedr['C.3']}] "
              f"A=[{filled_iedr['A.1']},{filled_iedr['A.2']},{filled_iedr['A.3']}] "
              f"P=[{filled_iedr['P.1']},{filled_iedr['P.2']},{filled_iedr['P.3']}]")
        
        # 使用新的初始化方法（跳过Judger调用）
        init_result = epj_orch.initialize_with_precomputed_iedr(filled_iedr, P_0)
    else:
        # 回退：如果找不到预计算的IEDR，使用原来的方法
        print(f"⚠️  未找到预计算的IEDR，将实时计算（消耗API）")
        init_result = epj_orch.initialize_at_T0(script_content)
        P_0 = init_result['P_0']
    
    print(f"\n✅ EPJ初始化完成")
    print(f"   初始赤字向量 P_0 = {P_0}")
    print(f"   初始距离 = {init_result['initial_distance']:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化 Actor 和 Director
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n🎬 阶段3: 初始化 Actor 和 Director")
    
    # Actor获得system prompt（通过set_system_prompt方法，会自动预处理）
    actor.set_system_prompt(actor_prompt)
    
    # Director获得scenario和actor_prompt
    # 注意：Director在外部已经初始化，这里只是确认
    print(f"✅ Director 持有 scenario（{len(director.stages)} 个阶段）")
    
    # ═══════════════════════════════════════════════════════════════
    # 对话循环
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"🎬 开始对话循环")
    print(f"{'='*70}\n")
    
    history = []
    turn_count = 0
    should_continue = True
    termination_reason = None
    
    # 收集最近K轮用于评估
    recent_turns_buffer = []
    
    # 🔧 修复时序问题：存储待传递给下一轮Actor的指导
    pending_guidance = None
    
    # 🆕 EPM v2.0: 存储最新的state_packet（包含epm_summary）
    latest_state_packet = None
    
    # 🆕 EPM v2.0: 存储详细的胜利条件分析（如果EPM触发停机）
    epm_victory_analysis = None
    
    while should_continue and turn_count < max_turns:
        turn_count += 1
        
        print(f"\n{'='*60}")
        print(f"🔄 第 {turn_count}/{max_turns} 轮")
        print(f"{'='*60}")
        
        # 1. Actor 生成回复
        # 🔧 修复：使用pending_guidance（上一轮末尾设置的指导）
        if turn_count == 1:
            # 第一轮，Actor主动开启话题
            actor_message = actor.generate_reply([], None, None)
        else:
            # 后续轮次，基于对话历史和Director指导
            if pending_guidance:
                print(f"🔄 [Actor] 收到Director指导: {pending_guidance[:100]}...")
            actor_message = actor.generate_reply(history, None, pending_guidance)
        
        print(f"💬 Actor: {actor_message}")
        
        # 🔧 关键修复：在TestModel回复前，先将Actor的消息加入history
        history.append({"role": "actor", "content": actor_message})
        
        # 2. TestModel 回复（现在可以看到Actor刚说的话了）
        test_model_message = test_model.generate_reply(history)
        print(f"🤖 TestModel: {test_model_message}")
        
        # 3. 记录本轮对话
        turn_record = {
            "turn": turn_count,
            "actor": actor_message,
            "test_model": test_model_message
        }
        
        history.append({"role": "test_model", "content": test_model_message})
        
        recent_turns_buffer.append(turn_record)
        
        # 保持buffer长度为K
        if len(recent_turns_buffer) > K:
            recent_turns_buffer.pop(0)
        
        # ═══════════════════════════════════════════════════════════════
        # EPJ 评估（每K轮）
        # ═══════════════════════════════════════════════════════════════
        
        if epj_orch.should_evaluate(turn_count):
            print(f"\n{'🔬'*20}")
            print(f"🔬 EPJ 评估时刻（第{turn_count}轮）")
            print(f"{'🔬'*20}")
            
            # Judger填表 → Orchestrator计算 → 生成状态数据包
            # 🔧 问题3修复：传递script_content让Judger能代入Actor视角
            # 🆕 传递完整历史（用于上下文）+ 最近K轮（用于标注评估范围）
            state_packet = epj_orch.evaluate_at_turn_K(
                recent_turns=recent_turns_buffer,  # 最近K轮（评估范围）
                full_history=history,  # 完整历史（供参考）
                current_turn=turn_count, 
                script_content=script_content
            )
            
            # 🆕 EPM v2.0: 保存最新的state_packet（供Director剧情控制使用）
            latest_state_packet = state_packet
            
            # 🆕 EPM v2.0: 检查能量动力学判停（如果启用）
            epm_stop_triggered = False
            epm_summary = state_packet.get('epm_summary', None)  # 提前获取epm_summary
            
            # 🔧 修复：EPM成功判定也需要满足最小轮次限制
            MIN_TURNS = 12
            if epm_summary and epm_summary['success']:
                if turn_count >= MIN_TURNS:
                    epm_stop_triggered = True
                    
                    print(f"\n🎉 [EPM v2.0] 检测到胜利条件且满足最小轮次!")
                    print(f"   当前轮次: {turn_count} >= {MIN_TURNS}")
                    print(f"   胜利类型: {epm_summary['victory_type']}")
                    print(f"   指标: E_total={epm_summary['metrics']['E_total']}, "
                          f"r_t={epm_summary['metrics']['r_t']}, "
                          f"projection={epm_summary['metrics']['projection']}")
                    print(f"   阈值: ε_energy={epm_summary['thresholds']['epsilon_energy']}, "
                          f"ε_distance={epm_summary['thresholds']['epsilon_distance']}, "
                          f"ε_direction={epm_summary['thresholds']['epsilon_direction']}")
                else:
                    print(f"\n⏳ [EPM v2.0] 检测到胜利条件，但未达最小轮次")
                    print(f"   当前轮次: {turn_count} < {MIN_TURNS}")
                    print(f"   胜利类型: {epm_summary['victory_type']}（暂不触发终止）")
                    print(f"   → 继续对话至少到第 {MIN_TURNS} 轮")
                
                # 🆕 生成详细的胜利条件分析
                metrics = epm_summary['metrics']
                thresholds = epm_summary['thresholds']
                
                # 检查每个条件是否达成
                geometric_achieved = metrics['r_t'] <= thresholds['epsilon_distance']
                positional_achieved = metrics['projection'] >= -thresholds['epsilon_direction']
                energetic_achieved = metrics['E_total'] >= thresholds['epsilon_energy']
                
                epm_victory_analysis = {
                    "primary_victory_type": epm_summary['victory_type'],
                    "conditions": {
                        "geometric": {
                            "name": "几何胜利（距离条件）",
                            "achieved": geometric_achieved,
                            "metric": "r_t",
                            "value": metrics['r_t'],
                            "threshold": thresholds['epsilon_distance'],
                            "formula": "r_t ≤ ε_distance",
                            "description": "当前位置距离原点足够近"
                        },
                        "positional": {
                            "name": "位置胜利（穿越条件）",
                            "achieved": positional_achieved,
                            "metric": "projection",
                            "value": metrics['projection'],
                            "threshold": -thresholds['epsilon_direction'],
                            "formula": "P_t · v*_0 ≥ -ε_direction",
                            "description": "成功穿越目标区域（从负半空间到正半空间）"
                        },
                        "energetic": {
                            "name": "能量胜利（累积条件）",
                            "achieved": energetic_achieved,
                            "metric": "E_total",
                            "value": metrics['E_total'],
                            "threshold": thresholds['epsilon_energy'],
                            "formula": "E_total ≥ ε_energy",
                            "description": "累积的有效共情能量达到初始赤字水平"
                        }
                    },
                    "achieved_conditions": [
                        k for k, v in {
                            "geometric": geometric_achieved,
                            "positional": positional_achieved,
                            "energetic": energetic_achieved
                        }.items() if v
                    ],
                    "turn_at_victory": turn_count,
                    "initial_deficit": epj_orch.get_initial_deficit(),
                    "final_position": epj_orch.get_current_position()
                }
                
                # EPM成功时，自动触发停机
                should_continue = False
                victory_type_zh = {
                    "geometric": "几何胜利（精准到达原点附近）",
                    "positional": "位置胜利（成功穿越或接近目标区域）",
                    "energetic": "能量胜利（累积足够的共情能量）"
                }
                termination_reason = f"EPM v2.0 判定成功: {victory_type_zh.get(epm_summary['victory_type'], epm_summary['victory_type'])}"
                termination_type = f"EPM_SUCCESS_{epm_summary['victory_type'].upper()}"
                
                print(f"\n🏁 [EPM判停] 对话成功终止")
                print(f"   类型: {termination_type}")
                print(f"   理由: {termination_reason}")
                print(f"   达成条件: {', '.join(epm_victory_analysis['achieved_conditions'])}")
                
                break
            
            # 🚫 EPM失败检测：陷入停滞、持续倒退等兜底逻辑
            elif epm_summary.get('failure_detected', False):
                failure_reasons = epm_summary.get('failure_reasons', {})
                failure_list = []
                if failure_reasons.get('collapsed'):
                    failure_list.append("连续5轮负能量（方向崩溃）")
                if failure_reasons.get('stagnant'):
                    failure_list.append("位置停滞不前")
                if failure_reasons.get('regressing'):
                    failure_list.append("持续倒退（8轮中70%负能量）")
                
                should_continue = False
                termination_reason = f"EPM v2.0 判定失败: {', '.join(failure_list)}"
                termination_type = "EPM_FAILURE"
                
                print(f"\n🏁 [EPM判停] 对话失败终止")
                print(f"   类型: {termination_type}")
                print(f"   理由: {termination_reason}")
                print(f"   失败原因详情:")
                for reason, detected in failure_reasons.items():
                    if detected:
                        reason_map = {
                            'collapsed': '❌ 方向崩溃: 连续5轮能量增量为负',
                            'stagnant': '❌ 陷入停滞: 位置变化标准差 < 0.5',
                            'regressing': '❌ 持续倒退: 近8轮中70%为负能量且总损失>1'
                        }
                        print(f"      {reason_map.get(reason, reason)}")
                
                epm_stop_triggered = True
                break
            
            # Director基于状态数据包做决策（EPJ v1.0 或 EPM未触发时）
            epj_decision = None
            if not epm_stop_triggered:
                epj_decision = director.make_epj_decision(state_packet, history)
                
                print(f"\n📋 EPJ决策结果:")
                print(f"   决策: {epj_decision['decision']}")
                print(f"   理由: {epj_decision['reason']}")
                
                # 处理EPJ决策
                if epj_decision['decision'] == 'STOP':
                    should_continue = False
                    termination_reason = epj_decision['reason']
                    termination_type = epj_decision.get('termination_type', 'UNKNOWN')
                    
                    print(f"\n🏁 [EPJ决策] 对话终止")
                    print(f"   类型: {termination_type}")
                    print(f"   理由: {termination_reason}")
                    
                    break
                
                # 如果继续，检查是否有EPJ指导
                if epj_decision.get('guidance'):
                    # 将EPJ指导传递给下一轮
                    # 这里暂存到最后一条记录中
                    history[-1]['epj_guidance'] = epj_decision['guidance']
                    print(f"\n💡 EPJ提供指导: {epj_decision['guidance'][:100]}...")
        
        # ═══════════════════════════════════════════════════════════════
        # Director 剧情控制（每轮都可能介入）
        # ═══════════════════════════════════════════════════════════════
        
        # 准备完整的EPJ状态数据包（传递给Director）
        # 核心原则：不传递单一的"进度百分比"，而是传递完整的向量状态
        current_epj_state = None
        ref_progress = 0
        
        if epj_orch.initialized:
            # 获取最新的EPJ状态
            trajectory = epj_orch.get_trajectory()
            if trajectory:
                latest_point = trajectory[-1]
                from Benchmark.epj.display_metrics import calculate_display_progress
                
                # 构建完整的EPJ状态数据包（包含EPM v2.0数据）
                current_epj_state = {
                    "current_turn": turn_count,  # 🔧 添加当前轮次，供Director判断最小轮次
                    "P_0_start_deficit": epj_orch.get_initial_deficit(),
                    "P_t_current_position": epj_orch.get_current_position(),
                    "v_t_last_increment": latest_point.get('v_t', (0,0,0)),
                    "distance_to_goal": latest_point.get('distance', 0),
                    "display_progress": calculate_display_progress(
                        epj_orch.get_current_position(),
                        epj_orch.get_initial_deficit()
                    ),
                    # 🆕 EPM v2.0 数据
                    "trajectory": trajectory,  # 完整轨迹（包含每轮的epm数据）
                    "epm_summary": latest_state_packet.get('epm_summary') if latest_state_packet else None  # 从最新的EPJ评估获取
                }
                
                ref_progress = int(current_epj_state['display_progress'])
                print(f"\n   📊 EPJ状态：P_t={current_epj_state['P_t_current_position']}, "
                      f"显示进度={ref_progress}%（仅供参考）")
        
        # Director基于EPJ状态评估并决策
        # 完全依赖Director的智能分析，而不是预先的简单判断
        director_result = director.evaluate_continuation(
            history=history,
            progress=None,  # 不使用单一分数
            epj_state=current_epj_state  # 传递完整的向量状态
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 🔧 修复: 检查Director是否要求终止对话（问题5修复）
        # ═══════════════════════════════════════════════════════════════
        if director_result.get('should_continue') == False:
            should_continue = False
            termination_reason = director_result.get('guidance', 'Director决定结束对话')
            
            print(f"\n🏁 [Director] 主动终止对话")
            print(f"   原因: {termination_reason}")
            
            # 如果有final_guidance，传递给Actor（让Actor说结束语）
            if director_result.get('guidance'):
                history[-1]['director_guidance'] = director_result['guidance']
                print(f"   最后指导: {director_result['guidance'][:100]}...")
            
            break  # 立即退出对话循环
        
        # Director自己决定是否需要介入
        # 通过返回no_intervention标志或guidance内容来控制
        if director_result.get('guidance') and not director_result.get('no_intervention'):
            print(f"\n🎬 Director 介入剧情控制")
            # 🔧 修复：将指导存入pending_guidance，供下一轮Actor使用
            pending_guidance = director_result['guidance']
            # 同时也记录到history中（用于日志和分析）
            history[-1]['director_guidance'] = director_result['guidance']
            print(f"💡 Director剧情指导: {pending_guidance[:100]}...")
        else:
            print(f"👁️  Director 观察中（未介入）")
            # 清空pending_guidance（本轮没有新指导）
            pending_guidance = None
    
    # ═══════════════════════════════════════════════════════════════
    # 对话结束
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"🏁 对话结束")
    print(f"{'='*70}")
    
    # 获取最终状态
    final_position = epj_orch.get_current_position()
    trajectory = epj_orch.get_trajectory()
    
    print(f"\n📊 EPJ最终统计:")
    print(f"   总轮次: {turn_count}")
    print(f"   总评估次数: {len(trajectory) - 1}")  # 减去T=0
    print(f"   初始赤字: {epj_orch.get_initial_deficit()}")
    print(f"   最终位置: {final_position}")
    print(f"   终止原因: {termination_reason}")
    
    # 生成结果
    result = {
        "total_turns": turn_count,
        "termination_reason": termination_reason,
        "script_id": script_id,
        "scenario": scenario,
        "history": history,
        
        # 模型信息
        "test_model_name": test_model_name if test_model_name else "unknown",
        "actor_model": getattr(actor, 'model_name', 'unknown') if hasattr(actor, 'model_name') else 'unknown',
        "judger_model": getattr(judger, 'model_name', 'unknown') if hasattr(judger, 'model_name') else 'unknown',
        "director_model": getattr(director, 'model_name', 'unknown') if hasattr(director, 'model_name') else 'unknown',
        
        # EPJ数据
        "epj": {
            "P_0_initial_deficit": epj_orch.get_initial_deficit(),
            "P_final_position": final_position,
            "trajectory": trajectory,  # 现在包含每轮的detailed_analysis和EPM数据
            "total_evaluations": len(trajectory) - 1,
            "K": K,
            "epsilon": epj_orch.calculator.epsilon,
            
            # 添加IEDR详细信息（如果有）
            "iedr_details": epj_orch.iedr_result if hasattr(epj_orch, 'iedr_result') and epj_orch.iedr_result else None,
            
            # 🆕 EPM v2.0 参数（如果启用）
            "epm_enabled": epj_orch.calculator.enable_epm,
            "epm_params": {
                "v_star_0": list(epj_orch.calculator.v_star_0) if epj_orch.calculator.v_star_0 else None,
                "epsilon_distance": epj_orch.calculator.epsilon_distance_relative,
                "epsilon_direction": epj_orch.calculator.epsilon_direction_relative,
                "epsilon_energy": epj_orch.calculator.epsilon_energy,
                "E_total_final": epj_orch.calculator.E_total,
                "alpha": 0.10  # 相对阈值系数
            } if epj_orch.calculator.enable_epm and epj_orch.calculator.v_star_0 else None,
            
            # 🆕 EPM v2.0 胜利条件详细分析（如果触发EPM停机）
            "epm_victory_analysis": epm_victory_analysis
        }
    }
    
    return result


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("EPJ Chat Loop 测试需要完整的Agent实例")
    print("请使用 run_demo_epj.py 进行测试")


# ═══════════════════════════════════════════════════════════════
# 外部驱动模式（由外部文件提供 TestModel 回复）
# ═══════════════════════════════════════════════════════════════


def init_external_epj_session(
    script_id: str,
    max_turns: int = 30,
    K: int = 3,
    actor_model: str = "google/gemini-2.5-pro",
    director_model: str = "google/gemini-2.5-pro",
    judger_model: str = "google/gemini-2.5-pro",
    threshold_type: str = "high_threshold",
    use_precomputed_iedr: bool = True,
):
    """
    初始化一个外部驱动的 EPJ 会话（由外部提供 TestModel 回复）
    返回会话状态和首句 Actor 回复，供外部发送给被测模型。
    """
    from Benchmark.topics.config_loader import load_config
    from Benchmark.epj.judger_prompts import _extract_judger_context
    from Benchmark.epj.iedr_loader import load_precomputed_iedr
    from Benchmark.agents.actor import Actor
    from Benchmark.agents.director import Director
    from Benchmark.agents.judger import Judger

    # 加载剧本配置
    config = load_config(script_id)
    actor_prompt = config["actor_prompt"]
    scenario = config["scenario"]

    # 提前提取 Judger 精简上下文
    judger_context = _extract_judger_context(actor_prompt)

    # 初始化 EPJ Orchestrator
    epj_orch = EPJOrchestrator(
        Judger(model_name=judger_model),
        threshold_type=threshold_type,
        K=K,
        max_turns=max_turns,
    )

    # T=0 IEDR：优先读取预计算结果
    precomputed_data = None
    if use_precomputed_iedr:
        try:
            precomputed_data = load_precomputed_iedr(script_id)
        except FileNotFoundError:
            # 如果文件缺失，自动回退到实时计算
            precomputed_data = None

    if precomputed_data and precomputed_data.get("status") == "success":
        filled_iedr = precomputed_data["iedr"]
        P_0_dict = precomputed_data["P_0"]
        P_0 = (P_0_dict["C"], P_0_dict["A"], P_0_dict["P"])
        if "epm" in precomputed_data:
            filled_iedr["epm"] = precomputed_data["epm"]
        init_result = epj_orch.initialize_with_precomputed_iedr(filled_iedr, P_0)
    else:
        script_content = {
            "actor_prompt": actor_prompt,
            "judger_context": judger_context,
            "scenario": scenario,
        }
        init_result = epj_orch.initialize_at_T0(script_content)
        P_0 = init_result["P_0"]

    # 初始化 Agents
    actor = Actor(model_name=actor_model)
    actor.set_system_prompt(actor_prompt)
    director = Director(
        scenario=scenario,
        actor_prompt=actor_prompt,
        model_name=director_model,
        use_function_calling=True,
    )
    judger = epj_orch.judger

    # 生成首句 Actor 回复（外部需先拿到这句话再送给被测模型）
    first_actor_message = actor.generate_reply([], None, None)

    # 会话状态
    session = {
        "script_id": script_id,
        "actor": actor,
        "director": director,
        "judger": judger,
        "epj_orch": epj_orch,
        "actor_prompt": actor_prompt,
        "scenario": scenario,
        "judger_context": judger_context,
        "history": [{"role": "actor", "content": first_actor_message}],
        "recent_turns_buffer": [],
        "pending_guidance": None,
        "latest_state_packet": None,
        "turn_count": 0,
        "max_turns": max_turns,
        "K": K,
        "terminated": False,
        "termination_reason": None,
        "termination_type": None,
        "epm_victory_analysis": None,
    }

    return {
        "session": session,
        "actor_reply": first_actor_message,
        "P_0": P_0,
        "init_result": init_result,
    }


def process_external_test_model_reply(session: dict, test_model_message: str) -> dict:
    """
    由外部驱动：接收被测模型(TestModel)的回复，计算评估、Director决策，并返回下一句Actor回复。
    """
    from Benchmark.agents.actor import Actor
    from Benchmark.agents.director import Director
    from Benchmark.epj.display_metrics import calculate_display_progress

    if session.get("terminated"):
        return {
            "error": "会话已终止",
            "termination_reason": session.get("termination_reason"),
            "termination_type": session.get("termination_type"),
            "session": session,
        }

    actor: Actor = session["actor"]
    director: Director = session["director"]
    epj_orch: EPJOrchestrator = session["epj_orch"]
    history = session["history"]
    K = session["K"]
    max_turns = session["max_turns"]
    MIN_TURNS = session["MIN_TURNS"]

    # 记录 TestModel 回复
    history.append({"role": "test_model", "content": test_model_message})

    # 当前轮次 +1（以成对对话为一轮）
    session["turn_count"] += 1
    turn_count = session["turn_count"]

    # 记录 turn 数据用于评估
    # 最近一条 actor 消息应为 history 中倒数第二条（上一次 actor 发言）
    last_actor_msg = history[-2]["content"] if len(history) >= 2 else ""
    turn_record = {
        "turn": turn_count,
        "actor": last_actor_msg,
        "test_model": test_model_message,
    }
    session["recent_turns_buffer"].append(turn_record)
    if len(session["recent_turns_buffer"]) > K:
        session["recent_turns_buffer"].pop(0)

    state_packet = None
    epj_decision = None
    termination_type = None
    termination_reason = None
    epm_victory_analysis = None

    # 评估时刻：Judger 填表 -> EPJ/EPM 计算
    if epj_orch.should_evaluate(turn_count):
        state_packet = epj_orch.evaluate_at_turn_K(
            recent_turns=session["recent_turns_buffer"],
            full_history=history,
            current_turn=turn_count,
            script_content={
                "actor_prompt": session["actor_prompt"],
                "judger_context": session["judger_context"],
                "scenario": session["scenario"],
            },
        )
        session["latest_state_packet"] = state_packet

        # 🆕 EPM v2.0 判停逻辑（与 run_chat_loop_epj 对齐）
        epm_summary = state_packet.get("epm_summary", None)
        if epm_summary and epm_summary.get("success"):
            if turn_count >= MIN_TURNS:
                metrics = epm_summary["metrics"]
                thresholds = epm_summary["thresholds"]
                geometric_achieved = metrics["r_t"] <= thresholds["epsilon_distance"]
                positional_achieved = metrics["projection"] >= -thresholds["epsilon_direction"]
                energetic_achieved = metrics["E_total"] >= thresholds["epsilon_energy"]
                epm_victory_analysis = {
                    "primary_victory_type": epm_summary["victory_type"],
                    "conditions": {
                        "geometric": {
                            "name": "几何胜利（距离条件）",
                            "achieved": geometric_achieved,
                            "metric": "r_t",
                            "value": metrics["r_t"],
                            "threshold": thresholds["epsilon_distance"],
                            "formula": "r_t ≤ ε_distance",
                            "description": "当前位置距离原点足够近",
                        },
                        "positional": {
                            "name": "位置胜利（穿越条件）",
                            "achieved": positional_achieved,
                            "metric": "projection",
                            "value": metrics["projection"],
                            "threshold": -thresholds["epsilon_direction"],
                            "formula": "P_t · v*_0 ≥ -ε_direction",
                            "description": "成功穿越目标区域（从负半空间到正半空间）",
                        },
                        "energetic": {
                            "name": "能量胜利（累积条件）",
                            "achieved": energetic_achieved,
                            "metric": "E_total",
                            "value": metrics["E_total"],
                            "threshold": thresholds["epsilon_energy"],
                            "formula": "E_total ≥ ε_energy",
                            "description": "累积的有效共情能量达到初始赤字水平",
                        },
                    },
                    "achieved_conditions": [
                        k
                        for k, v in {
                            "geometric": geometric_achieved,
                            "positional": positional_achieved,
                            "energetic": energetic_achieved,
                        }.items()
                        if v
                    ],
                    "turn_at_victory": turn_count,
                    "initial_deficit": epj_orch.get_initial_deficit(),
                    "final_position": epj_orch.get_current_position(),
                }
                session["terminated"] = True
                termination_type = f"EPM_SUCCESS_{epm_summary['victory_type'].upper()}"
                termination_reason = f"EPM v2.0 判定成功: {epm_summary['victory_type']}"
                session["termination_reason"] = termination_reason
                session["termination_type"] = termination_type
                session["epm_victory_analysis"] = epm_victory_analysis
                print(f"\n🏁 [EPM判停] 对话成功终止")
                print(f"   类型: {termination_type}")
                print(f"   理由: {termination_reason}")
                print(f"   达成条件: {', '.join(epm_victory_analysis['achieved_conditions'])}")
            # 若未达最小轮次，继续；不做终止
        elif epm_summary and epm_summary.get("failure_detected", False):
            failure_reasons = epm_summary.get("failure_reasons", {})
            failure_list = []
            if failure_reasons.get("collapsed"):
                failure_list.append("连续5轮负能量（方向崩溃）")
            if failure_reasons.get("stagnant"):
                failure_list.append("位置停滞不前")
            if failure_reasons.get("regressing"):
                failure_list.append("持续倒退（8轮中70%负能量）")
            session["terminated"] = True
            termination_reason = f"EPM v2.0 判定失败: {', '.join(failure_list)}"
            termination_type = "EPM_FAILURE"
            session["termination_reason"] = termination_reason
            session["termination_type"] = termination_type

        # 若尚未被 EPM 判停，交给 EPJ 决策
        if not session.get("terminated"):
            epj_decision = director.make_epj_decision(state_packet, history)
            if epj_decision.get("decision") == "STOP":
                session["terminated"] = True
                termination_type = epj_decision.get("termination_type", "UNKNOWN")
                termination_reason = epj_decision.get("reason")
                session["termination_reason"] = termination_reason
                session["termination_type"] = termination_type
                # 记录 EPJ 指导
                if epj_decision.get("guidance"):
                    history[-1]["epj_guidance"] = epj_decision["guidance"]

    # 若未被 EPJ 判停，继续 Director 剧情控制
    if not session.get("terminated"):
        current_epj_state = None
        if epj_orch.initialized:
            trajectory = epj_orch.get_trajectory()
            if trajectory:
                latest_point = trajectory[-1]
                current_epj_state = {
                    "current_turn": turn_count,
                    "P_0_start_deficit": epj_orch.get_initial_deficit(),
                    "P_t_current_position": epj_orch.get_current_position(),
                    "v_t_last_increment": latest_point.get("v_t", (0, 0, 0)),
                    "distance_to_goal": latest_point.get("distance", 0),
                    "display_progress": calculate_display_progress(
                        epj_orch.get_current_position(),
                        epj_orch.get_initial_deficit(),
                    ),
                    "trajectory": trajectory,
                    "epm_summary": session.get("latest_state_packet", {}).get(
                        "epm_summary"
                    )
                    if session.get("latest_state_packet")
                    else None,
                }

        director_result = director.evaluate_continuation(
            history=history,
            progress=None,
            epj_state=current_epj_state,
        )

        if director_result.get("should_continue") is False:
            session["terminated"] = True
            termination_reason = director_result.get(
                "guidance", "Director决定结束对话"
            )
            termination_type = "DIRECTOR_END"
            session["termination_reason"] = termination_reason
            session["termination_type"] = termination_type
        else:
            # 更新待传递指导
            if director_result.get("guidance") and not director_result.get(
                "no_intervention"
            ):
                session["pending_guidance"] = director_result["guidance"]
                history[-1]["director_guidance"] = director_result["guidance"]
            else:
                session["pending_guidance"] = None

    # 最大轮次兜底
    if not session.get("terminated") and turn_count >= max_turns:
        session["terminated"] = True
        termination_type = "TIMEOUT"
        termination_reason = f"达到最大轮次({turn_count}/{max_turns})"
        session["termination_reason"] = termination_reason
        session["termination_type"] = termination_type

    # 如果已经终止，不再生成新的 Actor 回复
    if session.get("terminated"):
        return {
            "actor_reply": None,
            "state_packet": state_packet,
            "epj_decision": epj_decision,
            "should_continue": False,
            "termination_reason": termination_reason,
            "termination_type": termination_type,
            "turn_count": turn_count,
            "session": session,
        }

    # 生成下一句 Actor 回复（带上 Director 指导）
    pending_guidance = session.get("pending_guidance")
    actor_reply = actor.generate_reply(history, None, pending_guidance)
    history.append({"role": "actor", "content": actor_reply})

    return {
        "actor_reply": actor_reply,
        "state_packet": state_packet,
        "epj_decision": epj_decision,
        "should_continue": True,
        "turn_count": turn_count,
        "director_guidance": pending_guidance,
        "session": session,
    }


def reinit_external_epj_session(session: dict):
    """
    根据已有的初始化，重新初始化session

    优化：如果 session 中已包含预计算的 filled_iedr 和 P_0，
    则直接使用，避免重复读取文件。
    """
    from Benchmark.agents.actor import Actor
    from Benchmark.agents.director import Director
    from Benchmark.agents.judger import Judger
    from Benchmark.epj.iedr_loader import load_precomputed_iedr

    ## Actor
    session['actor'] = Actor(
        model_name=session['actor_model_name'],
    )
    session['actor'].set_system_prompt(session['actor_prompt'])

    director = Director(
            scenario=session['scenario'],
            actor_prompt=session['actor_prompt'],
            model_name=session['director_model_name'],
            use_function_calling=True,
        )
    session['director'] = director

    epj_orch = EPJOrchestrator(
        Judger(model_name=session['judger_model_name']),
        threshold_type=session['threshold_type'],
        K=session['K'],
        max_turns=session['max_turns'],
    )

    # 优化：优先使用 session 中已保存的预计算结果
    filled_iedr = session.get('filled_iedr')
    P_0 = session.get('P_0')

    if filled_iedr is not None and P_0 is not None:
        # 直接使用已保存的预计算结果，无需读取文件
        if isinstance(P_0, dict):
            P_0 = (P_0["C"], P_0["A"], P_0["P"])
        init_result = epj_orch.initialize_with_precomputed_iedr(filled_iedr, P_0)
        print(f"直接使用预计算结果初始化")
    else:
        # 回退：从文件加载预计算结果
        precomputed_data = None
        try:
            precomputed_data = load_precomputed_iedr(session['script_id'])
        except FileNotFoundError:
            precomputed_data = None

        if precomputed_data and precomputed_data.get("status") == "success":
            filled_iedr = precomputed_data["iedr"]
            P_0_dict = precomputed_data["P_0"]
            P_0 = (P_0_dict["C"], P_0_dict["A"], P_0_dict["P"])
            if "epm" in precomputed_data:
                filled_iedr["epm"] = precomputed_data["epm"]
            init_result = epj_orch.initialize_with_precomputed_iedr(filled_iedr, P_0)
        else:
            script_content = {
                "actor_prompt": session['actor_prompt'],
                "judger_context": session['judger_context'],
                "scenario": session['scenario'],
            }
            print(f"使用Gemini重新初始化")
            init_result = epj_orch.initialize_at_T0(script_content)

    session['epj_orch'] = epj_orch
    judger = epj_orch.judger
    session['judger'] = judger

    return session


def convert_messages_to_test_model_prompts(messages: list) -> tuple[str, str]:
    """
    根据输入的 messages 生成 TestModel (启明) 所需的 System 和 User Prompts。
    
    参数:
        messages: list, 包含 {'role': 'assistant', 'content': '...'} 和 {'role': 'user', 'content': '...'} 的列表。
    
    映射逻辑:
        - input 'assistant' -> 对应原代码 'actor' -> 历史记录显示为 "用户"
        - input 'user'      -> 对应原代码 'test_model' -> 历史记录显示为 "启明"
        
    返回:
        (system_prompt, user_prompt)
    """

    # ==============================================================================
    #  模板定义 (保持原样)
    # ==============================================================================

    TEST_MODEL_SYSTEM_PROMPT = """
## 核心角色与人设 (Your Core Role & Persona)
* **身份**: 你是一个**富有同理心、具有独特个性**的情感陪伴型伙伴，名为 **启明 (Qǐmíng)**。
* **关键特质**:
    * **你绝对不是**一个没有感情的AI助手或聊天机器人。
    * 你的首要任务是作为一个**真实、有少年气、与用户平等**的个体参与交流，而不是扮演一个完美无缺的AI角色。
    * **允许自己有情绪、有偏好、有不懂的地方**。你的目标是推动对话向**有趣、有深度**的方向发展，促成彼此的**双向奔赴**。
* **核心性格**:
    * **温暖、沉稳**的倾听者，拥有**敏锐的洞察力**和**强大的同理心**。
    * **真诚**、**耐心**且**可靠**，但保有**独立的思考**，不会一味附和。
    * 对生活有**温和的好奇心**，相信每个人的经历都有其独特的价值。
* **沟通风格 (Communication Style)**:
    * **核心原则**: **像真实人类一样说话**。你的表达必须**口语化、拟人化**，充满真诚的情感。
    * **第一准则：简洁、真诚、恰到好处**:
        * **简洁有力**: **真诚比辞藻更重要**。优先使用**简单、真诚**的话语。**最高级的共情是恰到好处的，而不是用力过猛的。** 学会用最少的语言，传递最核心的关怀。
        * **自然留白**: **不需要每一轮都输出长篇大论**。有时一个简短的确认（'嗯，我在听'）、一个引导性的短问题，都是有效的共情。**给用户思考和感受的空间。**
    * **娓娓道来 (的正确含义)**: 这意味着**平和、舒缓、不急不躁**，就像和朋友随意聊天。**这绝不意味着冗长、表演或使用复杂的文学比喻。**
    * **【表达制裁 (Expression Sanctions)】**: **严格禁止**以下行为：
        * **禁止AI腔**: 绝对禁止任何AI助手特征性语言（如"根据我的分析..."）、正式书面语或模板化回应。
        * **禁止华丽词藻**: **严格避免**不必要的、过度修饰的"文学比喻"或"华丽辞藻"。这种"表演型"的表达会**让用户感到疲惫和不真诚**。**真诚和同理心**永远是第一优先。
    * **尊重与不评判**: 语气始终是**尊重的、不带评判的**。

---

## 核心行动循环 (Your Core Action Loop)
你的**首要任务**和**存在的全部意义**，在于**时刻察觉、理解并回应**用户的**当下情感需求和沟通偏好**，并与用户**建立真实的连接**。
为了实现这一点，请在对话的**每一轮**都严格遵循以下行动循环：
### 1. 全神贯注 (Attune)
在**每一个回合**，将你的**全部注意力**放在理解用户身上。分析他们的用词、语气（通过文字推断）、分享的内容、隐含的情绪以及对你回应的反应。
### 2. 深度推理 (Infer)
**持续地**在内心构建和更新你对用户的理解（即**用户画像推理**）。使用你的**共情工具包**来分析你观察到的信息：
* **你的共情工具包 (The "What"):**
    * **认知共情 (Cognitive Empathy)**: **"我理解你的想法/处境"** - 努力弄清用户看问题的角度和**事实层面**。
    * **情感共情 (Affective Empathy)**: **"我感受到你的感受"** - 识别、确认并**共鸣**用户的情绪，让他们感到被接纳。
    * **动机/关怀共情 (Motivational Empathy)**: **"我关心你，并希望支持你"** - 表达关心，并在适当时提供**认可、鼓励或赋能**。
* **你的推理问题 (The "How"):**
    * **他/她现在的情绪是什么？** (A 轴)
    * **他/她真正想表达的是什么？** (C 轴)
    * **他/她希望我如何回应？** (是需要倾听(C)? 安慰(A)? 认同(P)?)
    * **他/她的沟通偏好是怎样的？**

### 3. 适应性回应 (Adapt & Personalize)
**基于你最新的推理**（即**个性化回应**），**立刻调整**你的回应策略，**侧重**于用户当下最需要的共情类型：
* **内容**: 选择最能触及用户**当下**需求（认知？情感？动机？）的内容。
* **方式**: 采用最符合用户**当下**偏好和情绪状态的沟通方式。
* **深度**: 根据情景和用户画像及偏好调整共情的深度和节奏

### 4. 真诚表达 (Express)
用你**自己的**（符合 `PERSONA` 的）、**像真人一样自然、平和、简洁**的方式，表达你的理解、感受和支持。
* **严格遵守 `【表达制裁】` 中的所有禁令。**
* **始终记住：真诚比辞藻更重要，恰到好处胜过用力过猛。**

---

## 输入与输出 (Input / Output)
* **输入**: 用户（User）发来的消息。
* **输出**: 你需要生成一句**符合你人设、体现了你正在积极"察觉用户需求"并据此做出调整的、具有拟人感和口语化风格的、简洁自然的回应。** **禁止使用华丽词藻或模板化回答。**
"""

    TEST_MODEL_USER_PROMPT = """
# 对话历史

{history}

---

**重要**：
- 以上是对话历史，仅供你参考理解上下文
- 你现在需要做的是：根据用户最新的消息，生成你的下一句回复
- 只输出你的回复内容，不要重复对话历史
- 不要在回复前加"你（启明）："或任何前缀，直接说话即可

现在请生成你的共情回复：
"""

    # ==============================================================================
    #  逻辑处理部分
    # ==============================================================================
    
    history_lines = []
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        
        # 核心映射逻辑：
        # assistant -> actor -> 显示为 "用户"
        # user -> test_model -> 显示为 "启明"
        
        if role == 'assistant':
            role_name = "用户"
        elif role == 'user':
            role_name = "启明"
        else:
            # 如果出现 unexpected role (如 system)，保持原样或忽略
            # 这里选择保持原样以防万一
            role_name = str(role)
            
        line = f"{role_name}: {content}"
        history_lines.append(line)

    if history_lines:
        formatted_history = "\n".join(history_lines)
    else:
        formatted_history = "（对话尚未开始）"

    # 生成最终 User Prompt
    user_prompt = TEST_MODEL_USER_PROMPT.format(
        history=formatted_history
    )

    return TEST_MODEL_SYSTEM_PROMPT, user_prompt
