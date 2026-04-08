#!/usr/bin/env python3
"""
test_judger.py

单独测试 Judger 组件，完全模拟训练时的初始化和调用流程。
用途：验证不同 API 配置下 Judger 返回值的格式稳定性。

使用方法：
    cd /nas/naifan/MultiTurnRL
    python test_judger.py
"""

import os
import sys
import copy
import json
from pathlib import Path

# =====================================================================
# 【用户配置区域】修改此处来测试不同的 Judger API
# =====================================================================
# JUDGER_BASE_URL = "http://your-api-endpoint/v1"
# JUDGER_API_KEY  = "YOUR_API_KEY"   # ← 替换为你的 API Key
# JUDGER_MODEL    = "MiniMax_M25"  # ← 替换为要测试的模型

# JUDGER_BASE_URL = "http://your-api-endpoint/v1"
# JUDGER_API_KEY  = "YOUR_API_KEY"   # ← 替换为你的 API Key
# JUDGER_MODEL    = "qwen3_235b_a22b_instruct_2507"  # ← 替换为要测试的模型

JUDGER_BASE_URL = "http://0.0.0.0:9777/v1"
JUDGER_API_KEY  = "EMPTY"   # ← 替换为你的 API Key
JUDGER_MODEL    = "minimax_m25"  # ← 替换为要测试的模型
# =====================================================================

# 注入环境变量（必须在 import api.py 之前设置）
os.environ["JUDGER_BASE_URL"] = JUDGER_BASE_URL
os.environ["JUDGER_API_KEY"]  = JUDGER_API_KEY
os.environ["ACTOR_BASE_URL"] = JUDGER_BASE_URL
os.environ["ACTOR_API_KEY"] = JUDGER_API_KEY
os.environ["DIRECTOR_BASE_URL"] = JUDGER_BASE_URL
os.environ["DIRECTOR_API_KEY"] = JUDGER_API_KEY
os.environ["USE_LOCAL"] = "False"  # 禁用本地模式，使用上方配置的 API
os.environ["EXP_NAME"] = "test_judger"

# 将 sandbox 目录加入 Python 路径
BENCHMARK_PATH = "/nas/naifan/MultiTurnRL/env/sandbox"
if BENCHMARK_PATH not in sys.path:
    sys.path.insert(0, BENCHMARK_PATH)

# =====================================================================
# 导入框架模块（在环境变量设置之后）
# =====================================================================
import numpy as np
import pandas as pd

from Benchmark.orchestrator.chat_loop_epj import (
    process_external_test_model_reply,
    reinit_external_epj_session,
)


def main():
    print("=" * 60)
    print("Judger 单独测试")
    print(f"  模型: {JUDGER_MODEL}")
    print(f"  API:  {JUDGER_BASE_URL}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: 读取训练数据集第一条
    # ------------------------------------------------------------------
    print("\n[Step 1] 读取训练数据集...")
    df = pd.read_parquet("/nas/naifan/MultiTurnRL/data/train_all_0_727.parquet")
    row = df.iloc[0]
    print(f"  数据集共 {len(df)} 条，使用第 0 条（script_id={row['script_id']}）")

    extra_info   = row["extra_info"]
    init_agents  = extra_info["interaction_kwargs"]["init_agents"]
    actor_reply  = init_agents["actor_reply"]   # 第一条 actor 消息

    # 深拷贝 session，避免污染原始数据
    session = copy.deepcopy(init_agents["session"])

    # ------------------------------------------------------------------
    # Step 2: 将要测试的模型写入 session
    # ------------------------------------------------------------------
    print("\n[Step 2] 配置 Judger 模型...")
    session["judger_model_name"] = JUDGER_MODEL

    # 训练时的参数（与 MultiTurnRLAgent 同步）
    session["max_turns"]  = 20   # 测试时随意设置，不影响 Judger 调用
    session["K"]          = 1    # K=1 保证第一轮就触发 Judger 评估
    session["MIN_TURNS"]  = 1

    print(f"  judger_model_name : {session['judger_model_name']}")
    print(f"  K (评估周期)       : {session['K']}")
    print(f"  script_id         : {session['script_id']}")

    # ------------------------------------------------------------------
    # Step 3: 调用 reinit_external_epj_session —— 完全模拟训练时 turn==0 的初始化
    # ------------------------------------------------------------------
    print("\n[Step 3] 初始化 session（模拟训练 turn==0）...")
    # parquet 读出的 list 字段实际是 numpy array，需转成 Python list
    for _key in ("history", "recent_turns_buffer", "P_0"):
        if _key in session and not isinstance(session[_key], list):
            session[_key] = list(session[_key])
    session = reinit_external_epj_session(session)
    print("  ✅ session 初始化完成")
    print(f"  Judger 实例: {session['judger']}")
    print(f"  Judger 模型: {session['judger'].model_name}")

    for _ in range(10):
        # ------------------------------------------------------------------
        # Step 4: 构造模拟的 assistant（test_model）回复
        # ------------------------------------------------------------------
        # 训练时 history 初始已包含第一条 actor 消息，
        # process_external_test_model_reply 期望接收 test_model 对 actor 第一条消息的回复。
        fake_model_reply = "我理解你的感受，有时候聊天确实让人觉得很累。"
        print(f"\n[Step 4] 模拟 test_model 回复: {fake_model_reply!r}")

        # ------------------------------------------------------------------
        # Step 5: 调用 process_external_test_model_reply —— 完全模拟训练时每轮的调用
        # ------------------------------------------------------------------
        print("\n[Step 5] 调用 process_external_test_model_reply（触发 Judger）...")
        result = process_external_test_model_reply(session, fake_model_reply)

        # ------------------------------------------------------------------
        # Step 6: 打印结果
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("调用结果")
        print("=" * 60)

        print(f"\n[actor_reply]\n  {result.get('actor_reply')}")
        print(f"\n[should_continue]\n  {result.get('should_continue')}")
        print(f"\n[termination_reason]\n  {result.get('termination_reason')}")

        state_packet = result.get("state_packet")
        if state_packet:
            print("\n[state_packet]")
            for k, v in state_packet.items():
                print(f"  {k}: {v}")
        else:
            print("\n[state_packet] 本轮未触发评估（state_packet 为 None）")
            print("  原因：turn_count % K != 0，或 K 配置问题")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


if __name__ == "__main__":
    main()
