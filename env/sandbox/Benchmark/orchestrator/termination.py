from .progress_tracker import ProgressTracker

# Benchmark/orchestrator/termination.py (更新版)

def check_termination(tracker, max_turns, target_progress, is_fully_recovered=False):
    """
    检查是否应该终止对话（多层判断）
    
    Args:
        tracker: ProgressTracker实例
        max_turns: 最大回合数
        target_progress: 目标分数
        is_fully_recovered: 是否完全恢复
    
    Returns:
        (should_terminate, reason): 布尔值和终止原因
    """
    current_turn = tracker.get_turn_count()
    current_progress = tracker.get_progress()
    
    # 优先级1：真正完全恢复
    if is_fully_recovered:
        return (True, "✅ 倾诉者情绪已完全恢复")
    
    # 优先级2：分数显著超标（防止虚高）
    if current_progress >= target_progress * 1.3:  # 130分
        return (True, f"✅ 情绪改善显著超标（{current_progress}/{target_progress}）")
    
    # 优先级3：达标且趋势稳定
    if current_progress >= target_progress:
        # 检查最近3次评估的趋势
        recent_trends = tracker.get_recent_trends(3)
        positive_count = recent_trends.count("好转")
        
        # 如果最近3次中有2次以上是"好转"，认为稳定
        if positive_count >= 2:
            return (True, f"✅ 情绪持续改善，达到目标（{current_progress}/{target_progress}）")
        else:
            # 达标但趋势不稳定，继续对话
            print(f"⚠️ [终止检查] 分数已达标，但趋势不稳定，继续对话...")
            return (False, "继续对话（分数达标但情绪不稳定）")
    
    # 优先级4：最大回合数
    if current_turn >= max_turns:
        return (True, f"⏱️ 已达到最大回合数（{current_turn}/{max_turns}）")
    
    # 优先级5：分数严重为负（中止对话）
    if current_progress < -20:
        return (True, f"❌ 情绪严重恶化，建议中止对话（{current_progress}/{target_progress}）")
    
    # 继续对话
    return (False, "对话继续")