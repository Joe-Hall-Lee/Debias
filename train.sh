#!/bin/bash
#SBATCH -N 1 # 指定 node 的数量
#SBATCH -G 3 # 需要使用多少 GPU，n 是需要的数量
#SBATCH -o train.log # 把输出结果 STDOUT 保存在哪一个文件
#SBATCH -w wxhd11
# nohup bash train.sh > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0,1,2
WANDB_MODE=offline torchrun --nproc_per_node=3 --master_port=20002 train.py \
    --model_name_or_path ./models/vicuna-7b \
    --model_type "llama" \
    --data_path /home/disk/huanghui/data/Superficial/judgelm_sampled_data_both.jsonl \
    --bf16 True \
    --output_dir ./output/vicuna-generation-judgelm \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
