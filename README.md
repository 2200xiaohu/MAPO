# MAPO: Multi-turn Advantage Policy Optimization

A framework for multi-turn dialogue reinforcement learning training, built on top of [verl](https://github.com/volcengine/verl). MAPO implements various advantage estimation algorithms for multi-turn conversational scenarios, including GRPO, REINFORCE++, and PPO-GAE variants.

## Overview

MAPO addresses the challenge of training language models for multi-turn dialogue through reinforcement learning. It introduces:

- **Multi-turn dense reward management**: Turn-level reward signals instead of only episode-level rewards
- **Multiple advantage estimation algorithms**: GRPO Multiturn, REINFORCE++ V2-V5, PPO-GAE, and their variants with KL regularization
- **Flexible interaction environment**: A sandbox-based benchmark environment for dialogue simulation with configurable actors, directors, and judgers
- **Scalable training**: Support for multi-GPU training with Ray, SGLang rollout, and FSDP

## Project Structure

```
.
├── MultiTurnRLAgent.py              # Core: Multi-turn RL interaction agent
├── MultiTurnDenseRewardManager.py   # Core: Dense reward manager for turn-level rewards
├── custom_reward.py                 # Custom reward function interface
├── test_judger.py                   # Judger component testing script
├── test_debug.py                    # Debug utility
│
├── config/
│   ├── interaction_config.yaml      # Interaction environment configuration
│   └── multi_turn_grpo.yaml         # Training configuration (Hydra)
│
├── env/sandbox/                     # Dialogue simulation environment
│   ├── Benchmark/                   # Benchmark framework
│   │   ├── agents/                  # Actor, Director, Judger, Test Model
│   │   ├── epj/                     # EPJ scoring and metrics
│   │   ├── llms/                    # LLM API interfaces
│   │   ├── orchestrator/            # Chat loop orchestration
│   │   ├── prompts/                 # Prompt templates
│   │   └── topics/                  # Topic configuration
│   ├── config/                      # API configuration
│   ├── runner/                      # Benchmark runners
│   └── scripts/                     # Analysis and evaluation scripts
│
├── docs/
│   └── advantage_algorithms_analysis.md  # Detailed algorithm comparison
│
├── data/                            # Training data directory (not included)
│
├── test.sh                          # Main training script (MAPO)
├── test_reinforce_pp.sh             # REINFORCE++ experiment
├── test_reinforce_pp_v2.sh          # REINFORCE++ V2
├── test_reinforce_pp_v3.sh          # REINFORCE++ V3
├── test_reinforce_pp_v4.sh          # REINFORCE++ V4
├── test_reinforce_pp_v5.sh          # REINFORCE++ V5 (recommended)
├── test_reinforce_pp_v5_kl.sh       # REINFORCE++ V5 with KL regularization
├── test_reinforce_pp_v5_ray.sh      # REINFORCE++ V5 with Ray backend
├── test_reinforce_pp_v5_ray_kl.sh   # REINFORCE++ V5 Ray + KL
├── test_reinforce_pp_potential.sh   # REINFORCE++ with potential-based shaping
├── test_reinforce_pp_v4_on_policy.sh # REINFORCE++ V4 on-policy
├── test_on_policy.sh                # On-policy training
├── test_ppo_gae_multiturn.sh        # PPO-GAE multi-turn
├── test_grpo_multiturn_gamma.sh     # GRPO with gamma discounting
├── test_grpo_multiturn_gamma_ray.sh # GRPO gamma with Ray backend
├── exp_test.sh                      # Experiment testing
├── original.sh                      # Original baseline script
└── sglang_deploy.sh                 # SGLang model deployment
```

## Prerequisites

- Python 3.10+
- CUDA-compatible GPUs (8x GPUs recommended)
- [verl](https://github.com/volcengine/verl) framework (custom fork with multi-turn support)
- [Ray](https://www.ray.io/) for distributed training
- [SGLang](https://github.com/sgl-project/sglang) for efficient rollout
- [SwanLab](https://swanlab.cn/) for experiment tracking (optional)

## Installation

1. Install the verl framework with multi-turn RL support:
```bash
# Clone and install the custom verl fork
git clone <your-verl-fork-url>
cd verl
pip install -e .
```

2. Install additional dependencies:
```bash
pip install ray sglang swanlab datasets
```

3. Set up the sandbox environment:
```bash
cd env/sandbox
pip install -r requirements.txt
cp env.example .env
# Edit .env with your API keys
```

## Data Preparation

Training data should be in Parquet format and placed in the `data/` directory. Each row should contain:
- `prompt`: The initial dialogue prompt
- `script_id`: Unique script identifier
- `extra_info`: Dictionary containing interaction initialization kwargs (init_agents, session, etc.)

## Usage

### Quick Start

1. **Configure the interaction environment** by editing `config/interaction_config.yaml`:
```yaml
interaction:
  - name: "multi_turn_rl"
    class_name: "verl.interactions.MultiTurnRLAgent.MultiTurnRLAgent"
    config:
      base_url: "http://127.0.0.1:8000"
      max_turns: 15
      min_turns: 8
      reward_type: "distance_diff"
      gamma: 0.99
```

2. **Run the main training script** (MAPO with REINFORCE++ V5):
```bash
bash test_reinforce_pp_v5.sh
```

### Available Experiments

Each shell script corresponds to a different algorithm variant. Key parameters are configured at the top of each script:

| Script | Algorithm | Description |
|--------|-----------|-------------|
| `test.sh` | GRPO | Main MAPO training with GRPO advantage |
| `test_reinforce_pp_v5.sh` | REINFORCE++ V5 | Recommended: with turn & batch normalization |
| `test_grpo_multiturn_gamma.sh` | GRPO + Gamma | GRPO with gamma discounting |
| `test_ppo_gae_multiturn.sh` | PPO-GAE | PPO with GAE for multi-turn |
| `test_reinforce_pp_v5_kl.sh` | REINFORCE++ V5 + KL | V5 with KL divergence regularization |
| `test_reinforce_pp_v5_ray.sh` | REINFORCE++ V5 + Ray | V5 with Ray distributed backend |

### Key Configuration Parameters

Parameters can be modified in the shell scripts:

```bash
# Model
MODEL_PATH="/path/to/base/model"        # Base model path (e.g., Qwen3-8B)
TRAIN_FILE="/path/to/train.parquet"      # Training data

# Training
train_prompt_bsz=16                      # Prompt batch size
train_prompt_mini_bsz=8                  # Mini batch size
n_resp_per_prompt=4                      # Responses per prompt
max_prompt_length=3000                   # Max prompt tokens
max_response_length=30000                # Max response tokens

# Algorithm
adv_estimator=grpo                       # Advantage estimator type
clip_ratio_low=0.2                       # PPO clip ratio (low)
clip_ratio_high=0.28                     # PPO clip ratio (high)
temperature=1                            # Sampling temperature

# Multi-turn
new_max_turns=15                         # Maximum dialogue turns
new_min_turns=8                          # Minimum dialogue turns
new_reward_type="distance_diff"          # Reward type
```

### Testing the Judger

To verify the judger component works correctly:
```bash
python test_judger.py
```

## Algorithm Details

See `docs/advantage_algorithms_analysis.md` for a detailed comparison of all implemented advantage estimation algorithms, including:
- GRPO Multiturn
- REINFORCE++ V2-V5
- PPO-GAE Multi-turn
- Gamma discounting variants

## Citation

If you find this work useful, please cite:

```bibtex
@misc{mapo2025,
  title={MAPO: Multi-turn Advantage Policy Optimization},
  author={xiaohu},
  year={2025},
  url={https://github.com/2200xiaohu/MAPO}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
