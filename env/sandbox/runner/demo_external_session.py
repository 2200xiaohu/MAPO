#!/usr/bin/env python3
"""
示例：外部驱动模式（由外部提供 TestModel 回复）

流程：
1. init_external_epj_session -> 获取会话状态与首句 Actor 回复（发送给被测模型）
2. 外部获得模型回复后，调用 process_external_test_model_reply(session, reply)
3. 每轮拿到新的 Actor 回复，再发给被测模型；若返回 should_continue=False 则对话结束

此示例使用一个简单的“假”TestModel（回显+固定文本）演示调用流程。
"""

from Benchmark.orchestrator.chat_loop_epj import (
    init_external_epj_session,
    process_external_test_model_reply,
)


def fake_test_model_reply(actor_msg: str, turn: int) -> str:
    """模拟被测模型的回复（请替换为真实模型调用）"""
    return f"收到你的话({turn})：{actor_msg[:40]}... 我理解你的感受，我们可以聊聊吗？"


def main():
    # 1) 初始化会话（选择剧本ID、模型配置等）
    init = init_external_epj_session(
        script_id="001",       # 剧本ID
        max_turns=10,          # 最大轮次
        K=2,                   # 每2轮评估一次
        actor_model="google/gemini-2.5-pro",
        director_model="google/gemini-2.5-pro",
        judger_model="google/gemini-2.5-pro",
    )

    session = init["session"]
    actor_msg = init["actor_reply"]
    print(f"[Actor 首句] {actor_msg}")

    # 2) 模拟对话循环：外部模型 -> EPJ -> 返回下一句 Actor
    for step in range(1, session["max_turns"] + 1):
        # 假装外部模型根据 Actor 的上一句生成回复
        model_reply = fake_test_model_reply(actor_msg, step)
        print(f"[TestModel 第{step}轮] {model_reply}")

        # 将模型回复送入 EPJ，获取下一句 Actor + 评估信息
        result = process_external_test_model_reply(session, model_reply)

        # 若终止，输出原因并结束
        if not result.get("should_continue", True):
            print(f"[对话结束] 原因: {result.get('termination_reason')}, 类型: {result.get('termination_type')}")
            break

        # 正常继续，取出下一句 Actor 回复
        actor_msg = result["actor_reply"]
        print(f"[Actor 回复] {actor_msg}")

        # 如当前轮触发评估，可查看 state_packet
        if result.get("state_packet"):
            sp = result["state_packet"]
            print(f"[评分] 距离: {sp.get('distance_to_goal')}, 在区间: {sp.get('is_in_zone')}")


if __name__ == "__main__":
    main()

