# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("multi_turn_dense")
class MultiTurnDenseRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        #breakpoint()
        """We will expand this function gradually based on the available datasets"""

        '''
        当前的data中，attention mask，prompts，response都只有一轮的
        '''

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            uid = data_item.non_tensor_batch["uid"]

            # 已经Pad到max tokens
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length] # 包括user和assistant


            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            reward = data_item.non_tensor_batch['reward_scores']['user_turn_rewards']

            # 处理异常 reward 值（-1000 表示该轮次获取 reward 失败）
            INVALID_REWARD_MARKER = -1000
            valid_rewards = [r for r in reward if r != INVALID_REWARD_MARKER]

            if len(valid_rewards) < len(reward):
                # 存在异常 reward，用该轨迹的有效 reward 平均值替换
                if len(valid_rewards) > 0:
                    avg_reward = sum(valid_rewards) / len(valid_rewards)
                else:
                    # 所有 reward 都失败了，使用中性值 0
                    avg_reward = 0

                logger.warning(f"⚠️ [RewardManager] uid: {uid} 检测到 {len(reward) - len(valid_rewards)} 个异常 reward (-1000)，使用平均值 {avg_reward:.4f} 替换")
                reward = [r if r != INVALID_REWARD_MARKER else avg_reward for r in reward]

            reward = [-1 * r for r in reward]
            logger.warning(f"reward: {reward}")
            
            
            # 根据response_mask取位置, 本身为1，后一个为0即assistant last token
            response_mask = data_item.batch["response_mask"] 
            if response_mask[-1] == 1:
                # 特殊情况，assisatnt最后一轮被截断了
                logger.warning(f"uid: {uid} 特殊情况，assisatnt最后一轮可能被截断了，response: {response_str}")
                seq_token_idx = [j-1 for j in range(1, valid_response_length) if response_mask[j-1] == 1 and response_mask[j] == 0]
                seq_token_idx.append(valid_response_length.item()-1)
            
            else:
                seq_token_idx = [j-1 for j in range(1, valid_response_length+1) if response_mask[j-1] == 1 and response_mask[j] == 0]

            # 去重 + 排序
            seq_token_idx = list(set(seq_token_idx))
            seq_token_idx.sort()

            reward_type = "dense" 

            if reward_type == "sparse":
                # 最后一轮reward
                reward_tensor[i, seq_token_idx[-1]] = reward[-1]

            elif reward_type == "dense":
                # 每一轮一个reward
                #assert len(seq_token_idx) == len(reward), f"seq_token_idx: {seq_token_idx}, reward: {reward}"
                if len(seq_token_idx) < len(reward):
                    logger.warning(f"uid: {uid} reward和response超级奇怪现象，seq_token_idx长度小于reward长度\nseq_token_idx: {seq_token_idx}, reward: {reward}\nprompt: {prompt_str}\n\nresponse: {response_str}\n")
                if len(seq_token_idx) > len(reward):
                    logger.warning(f"uid: {uid} reward小于seq_token_idx长度\nseq_token_idx: {seq_token_idx}, reward: {reward}\nprompt: {prompt_str}\n\nresponse: {response_str}\n")
                    # 生成到达max length被中断，所以不会被计算reward
                    seq_token_idx = seq_token_idx[:len(reward)]
                    
                for idx, r in zip(seq_token_idx, reward):
                    reward_tensor[i, idx] = r

            #reward_tensor[i, seq_token_idx[-1]] = reward[-1] # 只取最后一轮的reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
