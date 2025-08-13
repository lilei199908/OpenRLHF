set -x
ROOT_PATH='/data1/lilei'
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain $ROOT_PATH/Qwen3-4B \
   --remote_rm_url /data1/lilei/OpenRLHF/examples/python/reward_func_aime2024.py \
   --ckpt_path $ROOT_PATH/test_scripts/ckpt/Qwen3-4B \
   --save_hf_ckpt \
   --micro_train_batch_size 10 \
   --train_batch_size 80 \
   --micro_rollout_batch_size 10 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 5 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data /data1/lilei/aime_2024/data/train-00000-of-00001.parquet \
   --input_key problem \
   --label_key answer \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \
