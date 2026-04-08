#!/usr/bin/env bash
cd /nas/naifan/verl

source /nas/miniconda3/etc/profile.d/conda.sh
# conda config --add envs_dirs /new_algorithm_nas/new_algorithm_nas/miniconda3/envs

conda init --all
conda env list
conda activate naifan_agent

swanlab login --relogin -k $SWANLAB_API_KEY

set -xeuo pipefail

cp -f /nas/naifan/MultiTurnRL/api_interaction.py /nas/naifan/verl/verl/interactions/api_interaction.py

#ray start --head --dashboard-host=0.0.0.0

project_name='agent'
exp_name='test'

CKPTS_DIR="/nas/naifan/MultiTurnRL/checkpoints/${project_name}/${exp_name}"

MODEL_PATH="/nas/models/Qwen/Qwen3-0.6B"
TRAIN_FILE="/nas/naifan/MultiTurnRL/data/demo_old.parquet"

CONFIG_PATH="/nas/naifan/MultiTurnRL/config"
TRAIN_BATCH_SIZE=4
MICRO_BATCH_SIZE=1
OFFLOAD=False


# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env=verl/trainer/runtime_env.yaml \
#     --no-wait \
#     -- \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_grpo_old' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=20000 \
    data.max_response_length=$((1024 * 3)) \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=["/nas/naifan/MultiTurnRL/data/demo_old.parquet"] \
    trainer.total_epochs=1 \
    custom_reward_function.path=/nas/naifan/MultiTurnRL/custom_reward_old.py \
    reward_model.reward_manager=custom_reward_dapo \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    data.return_multi_modal_inputs=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=/nas/naifan/MultiTurnRL/config/interaction_config_old.yaml \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.name=sglang \
    #actor_rollout_ref.rollout.multi_turn.tool_config_path="/data/home/user0/algorithm_nas/algorithm_nas/policy_model/agent/config/chat_api_tool_config.yaml" \
    #data.enable_thinking=True \
    #+data.apply_chat_template_kwargs.enable_thinking=True \
    #actor_rollout_ref.rollout.mode=async \
