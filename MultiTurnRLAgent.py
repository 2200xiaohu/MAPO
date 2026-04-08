# my_project/interactions/external_api.py
import os
import asyncio
import random
from typing import Any, Optional
from uuid import uuid4
import ast

from openai import AsyncOpenAI
import asyncio
from functools import partial

from .base import BaseInteraction
import sys
from pathlib import Path
import json
# 手动指定 Benchmark 包所在的目录
BENCHMARK_PATH = "/nas/naifan/MultiTurnRL/env/sandbox"

benchmark_dir = Path(BENCHMARK_PATH).resolve()
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from Benchmark.orchestrator.chat_loop_epj import (
    process_external_test_model_reply,
    reinit_external_epj_session
)
import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


import json
import os
import numpy as np
from functools import partial

# 1. [辅助工具] 处理 Numpy 数据类型 (float32, int64 等)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# 2. [核心逻辑] 覆盖写入单个 Instance 的文件
def save_instance_state(base_dir: str, instance_id: str, data: dict, exp_name: str):
    # 确保子文件夹存在
    record_dir = os.path.join(base_dir, exp_name)
    os.makedirs(record_dir, exist_ok=True)
    
    file_path = os.path.join(record_dir, f"{instance_id}.json")
    
    try:
        # 'w' 模式会清空旧内容写入新内容，实现替换效果
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NpEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Log Error] Save failed for {instance_id}: {e}")


def calculate_distance_diff(current_P_t, delta_vector, epsilon: float = 1.0, axis_weights: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    计算距离指标的变化量。
    
    逻辑：
    1. 根据当前坐标和变化向量(delta)，反推上一时刻坐标。
    2. 分别计算 Current_Dist 和 Previous_Dist。
    3. 返回 Diff = Current_Dist - Previous_Dist。
    
    Args:
        current_P_t: 当前的位置 (C, A, P)
        delta_vector: 相比于上次的位移向量 (dC, dA, dP) -> (Current - Previous)
        epsilon: 目标区域边界
        
    Returns:
        float: 距离的变化值。
               - 负数表示距离缩小了（情况变好了）
               - 正数表示距离变大了（情况变差了）
               - 0 表示没有有效变化（可能是在安全区内移动，或者刚好抵消）
    """
    
    # 1. 定义内部辅助函数：计算单点距离 (复用之前的逻辑)
    def _get_distance(coords):
        c, a, p = coords
        # 计算各维度距离分量
        c_d = (-epsilon - c) if c < -epsilon else 0
        a_d = (-epsilon - a) if a < -epsilon else 0
        p_d = (-epsilon - p) if p < -epsilon else 0
        # 返回欧氏距离
        wc, wa, wp = axis_weights # 解包权重
        weighted_sq_sum = wc*(c_d)**2 + wa*(a_d)**2 + wp*(p_d)**2
        return weighted_sq_sum ** 0.5

    # 2. 解包输入
    curr_c, curr_a, curr_p = current_P_t
    dc, da, dp = delta_vector

    # 3. 反推上一时刻坐标 (Previous = Current - Delta)
    prev_P_t = (curr_c - dc, curr_a - da, curr_p - dp)

    # 4. 分别计算距离
    dist_current = _get_distance(current_P_t)
    dist_previous = _get_distance(prev_P_t)

    # 5. 返回差值
    return dist_current - dist_previous


def calculate_potential_based_reward(current_P_t, delta_vector, gamma: float = 0.99, alpha: float = 1.0) -> float:
    """
    计算基于势能函数的 Reward (Potential-Based Reward Shaping)。
    
    公式：
    - 状态度量 D_t = ||(x, y, z)||_2  (欧氏距离)
    - 势能函数 Phi(s) = alpha / (alpha + D)
    - Reward r_t = gamma * Phi(s_t) - Phi(s_{t-1})
    
    Args:
        current_P_t: 当前的位置 (C, A, P)
        delta_vector: 相比于上次的位移向量 (dC, dA, dP) -> (Current - Previous)
        gamma: 折扣因子，默认 0.99
        alpha: 势能函数参数，控制梯度敏感区域，建议设为 D_initial_avg / 2
        
    Returns:
        float: 基于势能的 reward 值
               - 正数表示向目标接近（势能增加）
               - 负数表示远离目标（势能减少）
    """
    # 1. 计算欧氏距离
    def _calc_euclidean_distance(coords):
        return (coords[0]**2 + coords[1]**2 + coords[2]**2) ** 0.5
    
    # 2. 解包当前坐标
    curr_c, curr_a, curr_p = current_P_t
    dc, da, dp = delta_vector
    
    # 3. 反推上一时刻坐标 (Previous = Current - Delta)
    prev_P_t = (curr_c - dc, curr_a - da, curr_p - dp)
    
    # 4. 计算距离
    D_current = _calc_euclidean_distance(current_P_t)
    D_prev = _calc_euclidean_distance(prev_P_t)
    
    # 5. 计算势能函数 Phi(s) = alpha / (alpha + D)
    Phi_current = alpha / (alpha + D_current)
    Phi_prev = alpha / (alpha + D_prev)
    
    # 6. 计算 Reward: r_t = gamma * Phi(s_t) - Phi(s_{t-1})
    reward = gamma * Phi_current - Phi_prev
    
    return reward


def get_context_reward(result: dict):
    '''
    提取所有对话和对应的Reward
    '''
    history = result['history']
    
    rewards = []
    for r in result['epj']['trajectory']:
        # 暂且将distance作为Reward
        rewards.append(r['distance'])
    return history, rewards

class MultiTurnRLAgent(BaseInteraction):
    """
    - 从 messages 中取最近一条 assistant 文本
    - 作为 user 输入请求外部 Chat Completions
    - 返回外部回复作为环境响应；仅用 max_rounds 控制终止
    - 本实现不计算回合奖励（恒 0.0）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        #breakpoint()
        print(f"config: {config}")
        self.max_turns = config.get('max_turns', -1)
        self.val_turns = config.get('val_turns', -1)
        self.min_turns = config.get('min_turns', 8)
        self.reward_type = config.get('reward_type', 'distance_to_goal') # distance_to_goal, distance_diff, Potential_Based_reward
        self.exp_name = config.get('exp_name', 'default')
        # 新增：用于 Potential_Based_reward 的参数
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.alpha = config.get('alpha', 1.0)   # 势能函数参数
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        #print(f"start_interaction: {instance_id} ")
        if instance_id is None:
            instance_id = str(uuid4())
        # 目标是重新得到scipt id，然后初始化session
        # 替换data数据中初始化的actor回复
        #breakpoint() 
        session = kwargs['init_agents']['session']
        session['max_turns'] = self.max_turns # 同步max_turns
        session['K'] = self.val_turns 
        session['MIN_TURNS'] = self.min_turns
        #logger.warning(f"kwargs: {kwargs}")   
        '''
        kwargs:
        {
            "init_agents": {
                "session": {
                    "actor": Actor,
                    "director": Director,
                    "epj_orch": EPJOrchestrator,
                    "judger": Judger,
                }
            }
        }
        '''
        self._instance_dict[instance_id] = {
            "session": session,
            "turn": 0,
            "max_turns": self.max_turns,
            "reward_history": []
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        #breakpoint()
        session = self._instance_dict[instance_id]["session"]
        turn = self._instance_dict[instance_id]["turn"]
        
        model_reply = messages[-1]['content']
        if len(model_reply) < 20:
            logger.warning(f"model_reply为空，instance_id: {instance_id}, turn: {turn}\nMessages:\n{messages}")
        # 获取当前的事件循环
        loop = asyncio.get_running_loop()
        #logger.warning(f"Here is instance id: {instance_id}, turn: {turn}\nMessages:\n{messages}\nModel Reply:\n{model_reply}")
        #logger.warning(f"Here is instance id: {instance_id}, last turn: {turn}\n Messages: {messages[-2:]}\n")
        if turn == 0:
            # 如果 reinit 也是耗时的，也需要放入 executor
            # session = reinit_external_epj_session(session) 
            session = await loop.run_in_executor(
                None,  # None 使用默认的 ThreadPoolExecutor
                reinit_external_epj_session,
                session
            )
        
        # [核心修改] 将同步阻塞函数放入线程池运行，并 await 结果
        # 这样主线程的 Event Loop 会立即释放去处理其他 Task
        result = await loop.run_in_executor(
            None,  # None 表示使用默认线程池
            partial(process_external_test_model_reply, session, model_reply)
        )
        # 更新session
        self._instance_dict[instance_id]["session"] = result['session']
        actor_msg = result["actor_reply"]
        #print(f"[Actor 回复] {actor_msg}")
        reward = None
        if result.get("state_packet"):
            sp = result["state_packet"]

            if self.reward_type == 'v_t_last_increment':
                increment_vector = ast.literal_eval(sp.get('v_t_last_increment'))
                reward = increment_vector[0] + increment_vector[1] + increment_vector[2]
                reward = -1 * reward # 取反，因为后面处理reward时再次取反了

            elif self.reward_type == 'distance_to_goal':
                reward = sp.get('distance_to_goal')
            
            elif self.reward_type == 'distance_diff':
                is_success = sp.get('epm_summary').get('success')
                if is_success:
                    success_turn = sp.get('epm_summary').get('turn')
                    
                increment_vector = ast.literal_eval(sp.get('v_t_last_increment'))
                current_P_t = ast.literal_eval(sp.get('P_t_current_position'))
                epsilon = sp.get('epsilon')
                reward = calculate_distance_diff(current_P_t, increment_vector, epsilon, axis_weights=(1.0, 1.0, 1.0)) # new distance - old distance
            
            elif self.reward_type == 'Potential_Based_reward':
                # 基于势能函数的 Reward Shaping
                increment_vector = ast.literal_eval(sp.get('v_t_last_increment'))
                current_P_t = ast.literal_eval(sp.get('P_t_current_position'))
                reward = calculate_potential_based_reward(
                    current_P_t, 
                    increment_vector, 
                    gamma=self.gamma, 
                    alpha=self.alpha
                )
                reward = -1 * reward # 取反，因为后面处理reward时再次取反了
            else:
                raise ValueError(f"Invalid reward type: {self.reward_type}")
            #print(f"[评分] 距离: {sp.get('distance_to_goal')}, 在区间: {sp.get('is_in_zone')}")
        self._instance_dict[instance_id]["turn"] += 1
        self._instance_dict[instance_id]["reward_history"].append(reward)
        
        should_stop = self._instance_dict[instance_id]["turn"] >= self._instance_dict[instance_id]["max_turns"] or not result.get("should_continue", False) or result['session'].get("terminated", False)

        # 设置默认惩罚值，-1000 将在 RewardManager 中被该轨迹的平均 reward 替换
        if reward is None:
            logger.warning(f"⚠️ [Reward] reward 为 None，设置默认惩罚值 -1000{result}should_stop: {should_stop}")
            reward = -1000

        if should_stop:
            logger.warning(f"[对话结束] 原因: {result.get('termination_reason')}, 类型: {result.get('termination_type')}")

        log_data = {
            "instance_id": instance_id,
            "current_turn": self._instance_dict[instance_id]["turn"],
            "finished": should_stop,
            "messages": messages, 
            "current_reward": reward,
            "reward_history": self._instance_dict[instance_id]["reward_history"],
            
            "model_reply": model_reply,
            "actor_reply": actor_msg,
            "debug_info": result.get("state_packet", {})
        }
        # 2. 异步调用保存函数
        # 路径设置为你的 log 根目录，会自动创建 records 子目录
        log_base_dir = "/nas/naifan/MultiTurnRL/log/"
        
        await loop.run_in_executor(
            None, 
            partial(save_instance_state, log_base_dir, instance_id, log_data, self.exp_name)
        )
        # ======================================================
        
        return should_stop, actor_msg, reward, {"result": result}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        '''
        好像没什么用
        '''
        return 100

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# # 简单的合并脚本示例
# import glob, json
# all_data = [json.load(open(f)) for f in glob.glob("/nas/.../log/records/*.json")]
