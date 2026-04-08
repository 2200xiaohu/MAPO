#!/usr/bin/env bash
cd /nas/naifan/verl_reinforce_pp

# source /nas/miniconda3/etc/profile.d/conda.sh
# # conda config --add envs_dirs /new_algorithm_nas/new_algorithm_nas/miniconda3/envs

# conda init --all
# conda env list
# conda activate naifan_agent

# pip install swanlab
# pip install datasets==4.1.1

swanlab login --relogin -k $SWANLAB_API_KEY

set -xeuo pipefail

cp -f /nas/naifan/MultiTurnRL/MultiTurnRLAgent.py /nas/naifan/verl_reinforce_pp/verl/interactions/MultiTurnRLAgent.py
cp -r /nas/naifan/MultiTurnRL/MultiTurnDenseRewardManager.py /nas/naifan/verl_reinforce_pp/verl/workers/reward_manager/MultiTurnDenseRewardManager.py

#ray start --head --dashboard-host=0.0.0.0

#rm -rf /nas/naifan/MultiTurnRL/checkpoints/multi_turn_rl

project_name='multi_turn_rl_reinforce_pp_v5'
exp_name='qwen3_32b_train_max_turns_20_dense_distance_diff_235b_moe_reinforce_pp_v5_turn_normalization_batch_normalization_final_kl'

adv_estimator=reinforce_plus_plus_multiturn_v5

use_kl_in_reward=True
kl_coef=0.05
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((3000))
max_response_length=$((25000))
enable_overlong_buffer=True
overlong_buffer_len=$((10000 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# 为了保证尽可能on-policy，train_prompt_bsz和train_prompt_mini_bs要相近
train_prompt_bsz=32
train_prompt_mini_bsz=16
train_prompt_micro_bsz=1
n_resp_per_prompt=4

# train_prompt_bsz=8
# train_prompt_mini_bsz=4
# train_prompt_micro_bsz=1
# n_resp_per_prompt=2

NNODES=4
NGPUS_PER_NODE=8

CKPTS_DIR="/nas/naifan/MultiTurnRL/checkpoints/${project_name}/${exp_name}"

MODEL_PATH="/nas/models/Qwen/Qwen3-32B-Dense"
TRAIN_FILE="/nas/naifan/MultiTurnRL/data/train_all_0_727.parquet"

CONFIG_PATH="/nas/naifan/MultiTurnRL/config"
CONFIG_NAME="multi_turn_grpo.yaml"

# 修改interaction_config
new_max_turns=20
new_val_turns=1
new_min_turns=12
new_reward_type="distance_diff"
INTERACTION_CONFIG_PATH="${CONFIG_PATH}/interaction_config.yaml"
sed -i "s/\(\"max_turns\": \)[0-9]*/\1$new_max_turns/" "$INTERACTION_CONFIG_PATH"
# --- 修改 val_turns (整数，带注释) ---
sed -i "s/\(\"val_turns\": \)[0-9]*/\1$new_val_turns/" "$INTERACTION_CONFIG_PATH"
# --- 修改 min_turns (整数) ---
sed -i "s/\(\"min_turns\": \)[0-9]*/\1$new_min_turns/" "$INTERACTION_CONFIG_PATH"
# --- 修改 reward_type (字符串，带注释) ---
sed -i "s/\(\"reward_type\": \"\)[^\"]*/\1$new_reward_type/" "$INTERACTION_CONFIG_PATH"
# --- 修改 exp_name (字符串) ---
sed -i "s/\(\"exp_name\": \"\)[^\"]*/\1${exp_name}/" "$INTERACTION_CONFIG_PATH"
echo "配置修改完成："
grep -E "max_turns|val_turns|min_turns|reward_type|exp_name" "$INTERACTION_CONFIG_PATH"

# Back up envieroment
mkdir -p "${CKPTS_DIR}/code_backup"

echo "Backing up source code to ${CKPTS_DIR}/code_backup"

# Run script
cp -r "/nas/naifan/MultiTurnRL/test_reinforce_pp_v5_ray_kl.sh" "${CKPTS_DIR}/code_backup"

# Interaction config
cp -r "${CONFIG_PATH}" "${CKPTS_DIR}/code_backup"

# Data
cp -r "${TRAIN_FILE}" "${CKPTS_DIR}/code_backup"

# Verl
cp -r "/nas/naifan/verl_reinforce_pp" "${CKPTS_DIR}/code_backup"

# Algorithm
temperature=1
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1

gamma=0.85

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
infer_ppo_max_token_len=$((10 * max_prompt_length + 10 * max_response_length))
gen_tp=4
fsdp_size=-1

# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env=verl/trainer/runtime_env.yaml \
#     --no-wait \
#     -- \
export HYDRA_FULL_ERROR=1

# echo "正在清理旧的 Ray 进程..."
# ray stop --force
# echo "正在启动新的 Ray 进程..."
# ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# 注入环境变量
TARGET_API_FILE="/nas/naifan/MultiTurnRL/env/sandbox/Benchmark/llms/api.py"
use_local=True
if [ -f "$TARGET_API_FILE" ]; then
    echo "正在向 $TARGET_API_FILE 注入环境变量 EXP_NAME=${exp_name} ..."
    # 1. 为了防止重复运行脚本导致文件头部堆积多行相同的代码，
    #    先删除文件中可能已经存在的包含 "EXP_NAME" 的注入行。
    sed -i "/os.environ\['EXP_NAME'\]/d" "$TARGET_API_FILE"
    sed -i "/os.environ\['USE_LOCAL'\]/d" "$TARGET_API_FILE"
    # 2. 在文件的第 1 行之前插入代码。
    #    这行代码会导入 os 并强行设置环境变量。
    sed -i "1i import os; os.environ['EXP_NAME'] = '${exp_name}'" "$TARGET_API_FILE"
    sed -i "1i import os; os.environ['USE_LOCAL'] = '${use_local}'" "$TARGET_API_FILE"

    # 打印前几行确认注入成功
    echo "注入完成，文件头部内容如下："
    head -n 3 "$TARGET_API_FILE"
else
    echo "❌ 错误：未找到目标文件 $TARGET_API_FILE，无法注入环境变量！"
    # 如果这个文件至关重要，建议在这里 exit 1
fi

# 初始化Log文件夹
LOG_DIR="/nas/naifan/MultiTurnRL/log/${exp_name}"
if [ -d "$LOG_DIR" ]; then
    # 存在则暴力删除 (rm -rf: 递归删除且不询问)
    rm -rf "$LOG_DIR"
    echo "已删除旧目录: $LOG_DIR"
fi
mkdir -p "$LOG_DIR"
echo "已重新创建空目录: $LOG_DIR"


LOG_FILE="/nas/naifan/MultiTurnRL/reinforce_pp_v5_ray.log"
{
/usr/bin/python3.10 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="${CONFIG_NAME}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=['/nas/naifan/MultiTurnRL/data/valid_demo.parquet'] \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.kl_penalty="kl" \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.nccl_timeout=10800 \
    algorithm.gamma=${gamma} \
    \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_micro_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_prompt_micro_bsz} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_prompt_micro_bsz} \
    \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.test_freq=2 \
    trainer.save_freq=2 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=20 \
    trainer.val_before_train=False \
    +trainer.adv_data_save_dir="${LOG_DIR}/adv_data" \
    +trainer.adv_flush_interval=32 \
    data.return_multi_modal_inputs=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    custom_reward_function.path=/nas/naifan/MultiTurnRL/custom_reward.py \
    reward_model.reward_manager=multi_turn_dense \
    #actor_rollout_ref.rollout.mode=async \
    #trainer.resume_mode=resume_path \
    #trainer.resume_from_path=/nas/naifan/MultiTurnRL/checkpoints/multi_turn_rl_reinforce_pp_v5/qwen3_14b_train_max_turns_15_dense_distance_diff_235b_moe_reinforce_pp_v5_turn_normalization_batch_normalization_final/global_step_28 \
    

} 2>&1 | tee "$LOG_FILE"
