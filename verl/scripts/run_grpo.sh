# run on 8xH100
# make sure your current working directory is the root of the project

set -x

export HYDRA_FULL_ERROR=1

# claim抓取
export CLAIM_SERVER=127.0.0.1:8000
export CLAIM_SERVER_PATH=Qwen2.5-32B-Instruct

# claim评估和citation评估，可以和前面claim抓取一致
export CHECK_SERVER=127.0.0.1:8001
export CHECK_SERVER_PATH=Qwen2.5-32B-Instruct

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_FILE="/root/output/DeepCiteFact/data/rl_train_data_filter_grpo.parquet"
TEST_FILE=$TRAIN_FILE
ACTOR_MODEL_PATH="/root/output/DeepCiteFact-SFT/DeepCiteFact-SFT-epoch4"

SAVE_PATH="/root/output/DeepCiteFact-GRPO-ckpts"

TOOL_CONFIG_PATH="/root/output/DeepCiteFact/verl/examples/sglang_multiturn/config/tool_config/custom_tool_config.yaml"

current_time=$(date +"%Y-%m-%d_%H:%M:%S")

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export DIST_INIT_METHOD="tcp://$MASTER_ADDR:$MASTER_PORT"

PROJECT_NAME="DeepCiteFact-GRPO"
EXPERIMENT_NAME="qwen3-8b-deepcitefact-grpo"


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    global_profiler.tool=torch_memory \
    global_profiler.save_path=./mem_snapshots \
    global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000 \
    global_profiler.global_tool_config.torch_memory.stack_depth=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.format=custom \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    reward_model.enable=False \
    reward_model.reward_manager=custom \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode="disable" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@ 2>&1 | tee grpo_log.txt

